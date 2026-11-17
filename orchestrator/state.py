from typing import Literal, TypedDict


class GraphState(TypedDict):
    architecture_plan: str
    backend_summary: str
    frontend_summary: str
    qa_report: str
    human_feedback: str
    human_retry_target: Literal["backend_dev", "frontend_dev", "none"]
    retry_target: Literal["backend_dev", "frontend_dev", "none"]
    iteration_count: int
    max_iterations: int
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
