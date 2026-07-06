from typing import Annotated, TypedDict, Literal
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
