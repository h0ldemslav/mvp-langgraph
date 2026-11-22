import json
import sys
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
    route_start,
)
from state import GraphState

RUNS_DIR = Path(__file__).resolve().parent / "runs"
MILESTONES = ["M2", "M3.1", "M3.2", "M3.3"]


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("architect", architect_node)
    builder.add_node("backend_dev", backend_dev_node)
    builder.add_node("frontend_dev", frontend_dev_node)
    builder.add_node("qa_reviewer", qa_reviewer_node)
    builder.add_node("human_review", human_review_node)

    builder.add_conditional_edges(START, route_start)
    builder.add_edge("architect", "human_review")
    builder.add_edge("backend_dev", "frontend_dev")
    builder.add_edge("frontend_dev", "qa_reviewer")

    builder.add_conditional_edges("qa_reviewer", route_after_qa)
    builder.add_conditional_edges("human_review", route_after_human)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],
    )


def save_metrics(milestone: str, thread_id: str, state: dict, duration_seconds: float) -> Path:
    """Uloží metriky spotřeby pro vyhodnocení M22."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = (
        RUNS_DIR
        / f"langgraph_{milestone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "milestone": milestone,
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_seconds / 60, 2),
        "total_llm_calls": state.get("total_llm_calls", 0),
        "total_input_tokens": state.get("total_input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0),
        "qa_iterations": state.get("iteration_count", 0),
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def run_milestone(app, milestone: str) -> None:
    """Spustí jednu izolovanou session pro zadaný milník a uloží metriky."""
    thread_id = f"langgraph-{milestone}-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: GraphState = {
        "current_milestone": milestone,
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

    print(f"\n=== SPUŠTĚNÍ MILNÍKU {milestone} ===")    
    start_time = time.time()
    state = app.invoke(initial_state, config)

    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            break

        if milestone != "M2":
            print("\nQA report:")
            print(snapshot.values.get("qa_report", "Bez reportu."))
        else:
            print("\nArchitekt dokončil návrh (docs/ARCHITECTURE.md, ADR.md, DESIGN.md, RTM.md).")

        human_feedback = input(
            f"\nMilník {milestone}: Napiš 'ok' pro schválení, nebo zadej pokyn k opravě (zaznamenej do deníku M20): "
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

        if milestone == "M2":
            human_retry_target = "architect"
        else:
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
    metrics_file = save_metrics(milestone, thread_id, state, elapsed)
    print(f"\nMilník {milestone} dokončen.")
    print(
        f"Souhrn M22 -> Čas: {round(elapsed / 60, 2)} min | "
        f"Volání LLM: {state.get('total_llm_calls', 0)} | "
        f"Tokeny In/Out: {state.get('total_input_tokens', 0)} / {state.get('total_output_tokens', 0)}"
    )
    print(f"Detailní log uložen do: {metrics_file}")


if __name__ == "__main__":
    app = build_graph()

    # Možnost spustit jen konkrétní milník (napr. `python main.py M3.1`) nebo postupně všechny
    if len(sys.argv) > 1:
        selected = sys.argv[1].strip().upper()
        if selected not in MILESTONES:
            raise ValueError(f"Neznámý milník '{selected}'. Povolené hodnoty: {MILESTONES}")
        milestones_to_run = [selected]
    else:
        milestones_to_run = MILESTONES

    for idx, current_milestone in enumerate(milestones_to_run):
        run_milestone(app, current_milestone)
        if idx < len(milestones_to_run) - 1:
            input(
                f"\nProveď Git commit pro {current_milestone} a zápis do intervention-log.csv. "
                f"Poté stiskni Enter pro spuštění milníku {milestones_to_run[idx + 1]}..."
            )
