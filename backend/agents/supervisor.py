from langchain_core.messages import HumanMessage, AIMessage
from backend.config import get_llm
from backend.graph.state import AgentState

SUPERVISOR_PROMPT = """You are a Supervisor Agent in a multi-agent research system.
Your job is to analyze the user's query and produce a clear, numbered, step-by-step research and analysis plan.

The plan will be executed by:
- A Researcher Agent (web search, data gathering)
- A Data Analyst Agent (Python code execution, calculations, charts)

Rules:
- Be specific about what data to search for and what calculations to perform.
- If the query involves numbers/metrics, explicitly ask the Analyst to calculate them.
- If a chart would add value, ask the Analyst to generate one.
- Output ONLY the numbered plan, no preamble.
"""


def supervisor_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0)
    query = state["original_query"]

    feedback_section = ""
    if state.get("critic_feedback"):
        feedback_section = f"\n\nPrevious attempt failed. Critic feedback:\n{state['critic_feedback']}\nRevise the plan to address this."

    response = llm.invoke([
        HumanMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"User Query: {query}{feedback_section}"),
    ])

    plan = response.content
    return {
        **state,
        "plan": plan,
        "messages": state["messages"] + [AIMessage(content=f"**Supervisor Plan:**\n{plan}", name="supervisor")],
    }
