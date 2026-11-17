import json
import time
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from nodes import (
    architect_node,
    backend_dev_node,
    frontend_dev_node,
    human_review_node,
    qa_reviewer_node,
    route_after_human,
    route_after_qa,
)
from state import GraphState

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("architect", architect_node)
    builder.add_node("backend_dev", backend_dev_node)
    builder.add_node("frontend_dev", frontend_dev_node)
    builder.add_node("qa_reviewer", qa_reviewer_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "architect")
    builder.add_edge("architect", "backend_dev")
    builder.add_edge("backend_dev", "frontend_dev")
    builder.add_edge("frontend_dev", "qa_reviewer")

    builder.add_conditional_edges("qa_reviewer", route_after_qa)
    builder.add_conditional_edges("human_review", route_after_human)

    memory = MemorySaver()
    # Zastavení po architektovi (Milník M2) a před finálním schválením (Milník M3)
    return builder.compile(
        checkpointer=memory,
        interrupt_after=["architect"],
        interrupt_before=["human_review"],
    )


def save_metrics(thread_id: str, state: dict, duration_seconds: float) -> Path:
    """Uloží metriky spotřeby pro vyhodnocení M22."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RUNS_DIR / f"{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "total_llm_calls": state.get("total_llm_calls", 0),
        "total_input_tokens": state.get("total_input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0),
        "qa_iterations": state.get("iteration_count", 0),
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


if __name__ == "__main__":
    app = build_graph()
    thread_id = "langgraph-run-1"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: GraphState = {
        "architecture_plan": "",
        "backend_summary": "",
        "frontend_summary": "",
        "qa_report": "",
        "human_feedback": "",
        "human_retry_target": "none",
        "retry_target": "none",
        "iteration_count": 0,
        "max_iterations": 3,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,
    }

    start_time = time.time()

    # 1. Běh do milníku M2 (Návrh architektury)
    print("Spouštím fázi návrhu (Architect)...")
    state = app.invoke(initial_state, config)
    print("\nArchitekt dokončil návrh (ARCHITECTURE.md). Zkontroluj soubor v repozitáři.")
    input("Stiskni Enter pro schválení milníku M2 a spuštění implementace...")

    # 2. Smyčka implementace a lidského schválení milníku M3
    state = app.invoke(None, config)
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            break

        print("\nQA report:")
        print(snapshot.values.get("qa_report", "Bez reportu."))
        human_feedback = input(
            "\nMilník M3: Napiš 'ok' pro schválení, nebo zadej pokyn k opravě (zaznamenej do deníku M20): "
        ).strip()

        if human_feedback.lower() == "ok":
            app.update_state(
                config,
                {"human_feedback": "", "human_retry_target": "none"},
            )
            state = app.invoke(None, config)
            break
        if not human_feedback:
            print("Zadej 'ok' nebo pokyn k opravě.")
            continue

        while True:
            target_choice = input(
                "Který agent má změnu provést? Zadej 'backend' nebo 'frontend': "
            ).strip().lower()
            if target_choice == "backend":
                human_retry_target = "backend_dev"
                break
            if target_choice == "frontend":
                human_retry_target = "frontend_dev"
                break
            print("Neplatný cíl. Zadej 'backend' nebo 'frontend'.")

        app.update_state(
            config,
            {
                "human_feedback": human_feedback,
                "human_retry_target": human_retry_target,
                "iteration_count": 0,
            },
        )
        state = app.invoke(None, config)

    elapsed = time.time() - start_time
    metrics_file = save_metrics(thread_id, state, elapsed)
    print(f"\nHotovo. Metriky M22 uloženy do: {metrics_file}")
