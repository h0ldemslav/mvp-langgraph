import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.2,
)

class AgentState(TypedDict):
    task: str
    plan: str
    total_input_tokens: int
    total_output_tokens: int

def architect_node(state: AgentState):
    response = llm.invoke(
        f"Navrhni stručnou strukturu 3 souborů pro úkol: {state['task']}"
    )
    usage = response.usage_metadata or {}
    return {
        "plan": response.content,
        "total_input_tokens": state.get("total_input_tokens", 0) + usage.get("input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + usage.get("output_tokens", 0),
    }

def developer_node(state: AgentState):
    print("\n[Developer] Schváleno člověkem, pokračuji v implementaci podle plánu...")
    return state

builder = StateGraph(AgentState)
builder.add_node("architect", architect_node)
builder.add_node("developer", developer_node)

builder.add_edge(START, "architect")
builder.add_edge("architect", "developer")
builder.add_edge("developer", END)

# MemorySaver + interrupt_before umožní zastavit graf před krokem developer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["developer"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-run-1"}}
    initial_input = {
        "task": "FastAPI endpoint pro registraci účastníka skautské akce",
        "plan": "",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    # 1. Spuštění do prvního milníku (zastaví se před developer_node)
    result = graph.invoke(initial_input, config)
    print("--- NÁVRH ARCHITEKTA ---")
    print(result["plan"])
    print(f"\nSpotřeba (In/Out tokeny): {result['total_input_tokens']} / {result['total_output_tokens']}")

    # 2. Simulace schválení milníku člověkem a pokračování
    input("\nStiskni Enter pro schválení milníku a pokračování...")
    graph.invoke(None, config)