from typing import Annotated, Any, TypedDict, Literal
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full conversation + agent messages
    messages: Annotated[list, add_messages]
    # Original user query, preserved throughout the run
    original_query: str
    # Supervisor's step-by-step plan
    plan: str
    # Raw research findings from the Researcher agent
    research_findings: str
    # Code + output produced by the Analyst agent
    analysis_result: str
    # Critic's verdict: "pass" or "fail"
    critic_verdict: Literal["pass", "fail", ""]
    # Feedback from Critic when verdict is "fail"
    critic_feedback: str
    # Number of revision loops to prevent infinite cycles
    revision_count: int
    # Base64-encoded chart image (if any) from the Analyst
    chart_image: str
    # Structured source URLs extracted from research output
    sources: list[dict[str, str]]
    # Lightweight run metadata for API/UI observability
    run_metrics: dict[str, Any]
    # Non-fatal issues surfaced to the user
    warnings: list[str]
    # Heuristic quality score from 0.0 to 1.0
    quality_score: float
    # User-selected controls for research depth, style, and visuals
    user_preferences: dict[str, Any]
