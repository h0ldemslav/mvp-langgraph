from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END

from models import QADecision
from state import GraphState
from tools import ARCHITECT_TOOLS, DEVELOPER_TOOLS, QA_TOOLS

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1,
)


def _load_agent_prompts(agent_name: str) -> tuple[str, str]:
    """Načte system.md a user.md pro daného agenta."""
    agent_dir = PROMPTS_DIR / agent_name
    system_prompt = (agent_dir / "system.md").read_text(encoding="utf-8")
    user_prompt = (agent_dir / "user.md").read_text(encoding="utf-8")
    return system_prompt, user_prompt


def _run_agent_node(
    system_prompt: str, user_prompt: str, tools: list
) -> tuple[str, int, int, int]:
    """Spustí ReAct agenta a sečte tokeny i volání LLM ze všech mezikroků."""
    agent = create_agent(llm, tools=tools, prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})

    in_tokens = 0
    out_tokens = 0
    calls = 0
    final_text = ""

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            calls += 1
            usage = msg.usage_metadata or {}
            in_tokens += usage.get("input_tokens", 0)
            out_tokens += usage.get("output_tokens", 0)
            if isinstance(msg.content, str) and msg.content:
                final_text = msg.content

    return final_text, in_tokens, out_tokens, calls


def architect_node(state: GraphState) -> dict:
    sys_prompt, user_prompt = _load_agent_prompts("architect")

    text, in_tok, out_tok, calls = _run_agent_node(
        sys_prompt, user_prompt, ARCHITECT_TOOLS
    )
    return {
        "architecture_plan": text,
        "total_input_tokens": state["total_input_tokens"] + in_tok,
        "total_output_tokens": state["total_output_tokens"] + out_tok,
        "total_llm_calls": state["total_llm_calls"] + calls,
    }


def backend_dev_node(state: GraphState) -> dict:
    sys_prompt, user_prompt = _load_agent_prompts("backend_dev")

    feedback_section = ""
    if state.get("qa_report") and (
        state["retry_target"] == "backend_dev"
        or state["human_retry_target"] == "backend_dev"
    ):
        feedback_section += f"\nPředchozí QA report k opravě:\n{state['qa_report']}\n"
    if state.get("human_feedback"):
        feedback_section += (
            f"\nZpětná vazba od lidského kontrolora milníku:\n{state['human_feedback']}\n"
        )
    user_prompt += f"\n{feedback_section}"

    text, in_tok, out_tok, calls = _run_agent_node(
        sys_prompt, user_prompt, DEVELOPER_TOOLS
    )
    return {
        "backend_summary": text,
        "human_feedback": "",
        "human_retry_target": "none",
        "total_input_tokens": state["total_input_tokens"] + in_tok,
        "total_output_tokens": state["total_output_tokens"] + out_tok,
        "total_llm_calls": state["total_llm_calls"] + calls,
    }


def frontend_dev_node(state: GraphState) -> dict:
    sys_prompt, user_prompt = _load_agent_prompts("frontend_dev")
    
    feedback_section = ""
    if state.get("qa_report") and (
        state["retry_target"] == "frontend_dev"
        or state["human_retry_target"] == "frontend_dev"
    ):
        feedback_section += f"\nPředchozí QA report k opravě:\n{state['qa_report']}\n"
    if state.get("human_feedback"):
        feedback_section += (
            f"\nZpětná vazba od kontrolora milníku:\n{state['human_feedback']}\n"
        )
    user_prompt += f"\n{feedback_section}"

    text, in_tok, out_tok, calls = _run_agent_node(
        sys_prompt, user_prompt, DEVELOPER_TOOLS
    )
    return {
        "frontend_summary": text,
        "human_feedback": "",
        "human_retry_target": "none",
        "total_input_tokens": state["total_input_tokens"] + in_tok,
        "total_output_tokens": state["total_output_tokens"] + out_tok,
        "total_llm_calls": state["total_llm_calls"] + calls,
    }


def qa_reviewer_node(state: GraphState) -> dict:
    sys_prompt, user_prompt = _load_agent_prompts("qa_reviewer")

    raw_report, in_tok, out_tok, calls = _run_agent_node(
        sys_prompt, user_prompt, QA_TOOLS
    )

    router_llm = llm.with_structured_output(QADecision)
    decision: QADecision = router_llm.invoke(
        "Na základě tohoto QA reportu rozhodni, zda vše prošlo nebo kdo má chybu opravit:\n\n"
        f"{raw_report}"
    )

    new_iterations = state["iteration_count"] + (0 if decision.passed else 1)
    target = "none" if decision.passed else decision.retry_target

    return {
        "qa_report": f"{decision.summary}\n\nDetaily:\n{raw_report}",
        "retry_target": target,
        "iteration_count": new_iterations,
        "total_input_tokens": state["total_input_tokens"] + in_tok,
        "total_output_tokens": state["total_output_tokens"] + out_tok,
        "total_llm_calls": state["total_llm_calls"] + calls + 1,
    }


def human_review_node(state: GraphState) -> dict:
    return state


def route_after_qa(
    state: GraphState,
) -> Literal["backend_dev", "frontend_dev", "human_review"]:
    if state["iteration_count"] >= state["max_iterations"]:
        return "human_review"
    if state["retry_target"] == "backend_dev":
        return "backend_dev"
    if state["retry_target"] == "frontend_dev":
        return "frontend_dev"
    return "human_review"


def route_after_human(
    state: GraphState,
) -> Literal["backend_dev", "frontend_dev", "__end__"]:
    if not state["human_feedback"]:
        return END
    if state["human_retry_target"] == "none":
        raise ValueError("Human feedback requires an explicit developer target.")
    return state["human_retry_target"]
