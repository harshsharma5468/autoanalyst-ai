import logging
import threading

from langgraph.graph import StateGraph, END
from backend.graph.state import AgentState
from backend.agents.supervisor import supervisor_node
from backend.agents.researcher import researcher_node
from backend.agents.analyst import analyst_node
from backend.agents.critic import critic_node
from backend.agents.compiler import compiler_node
from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)
MAX_REVISIONS = 3


def route_after_critic(state: AgentState) -> str:
    """Route back to researcher if critic fails, else compile the final report."""
    if state["critic_verdict"] == "pass" or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "compiler"
    return "researcher"


def _build_graph_structure() -> StateGraph:
    """Build and return a compiled StateGraph (no checkpointer attached yet)."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst",    analyst_node)
    builder.add_node("critic",     critic_node)
    builder.add_node("compiler",   compiler_node)

    builder.set_entry_point("supervisor")
    builder.add_edge("supervisor", "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst",    "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"researcher": "researcher", "compiler": "compiler"},
    )
    builder.add_edge("compiler", END)
    return builder


def _try_postgres_checkpointer(conn_string: str, timeout: int = 8):
    """
    Attempt to create and set up a PostgresSaver.
    Runs in a daemon thread so it cannot block the server indefinitely.
    Returns the checkpointer on success, None on any failure or timeout.
    """
    result: list = []

    def _setup():
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            cp = PostgresSaver.from_conn_string(conn_string)
            cp.setup()
            result.append(cp)
        except Exception as exc:
            logger.warning("PostgresSaver setup failed: %s", exc)

    t = threading.Thread(target=_setup, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result:
        logger.info("LangGraph PostgresSaver connected successfully.")
        return result[0]

    logger.warning(
        "PostgresSaver not available (timeout or error). "
        "Running without state persistence."
    )
    return None


def build_graph(use_persistence: bool = True):
    builder = _build_graph_structure()

    if use_persistence and DATABASE_URL:
        checkpointer = _try_postgres_checkpointer(DATABASE_URL)
        if checkpointer:
            return builder.compile(checkpointer=checkpointer)

    return builder.compile()


# NOTE: Do NOT call build_graph() here at module level.
# main.py calls it lazily on first /analyze request via _get_graph().
# This prevents a hanging PostgreSQL connection from blocking server startup.
