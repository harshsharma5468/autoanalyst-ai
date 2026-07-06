from langchain_core.messages import HumanMessage, AIMessage
from backend.config import get_llm
from backend.tools.search import tavily_tool
from backend.graph.state import AgentState

RESEARCHER_PROMPT = """You are a Researcher Agent. You have access to a web search tool.
Given a research plan, execute the relevant search steps to gather real-time data, statistics, and reports.

Instructions:
- Run multiple targeted searches to cover all research steps in the plan.
- Synthesize findings into a structured report with sources cited.
- Include specific numbers, dates, and company names where found.
- Be factual. Do not invent data.
"""


def researcher_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0).bind_tools([tavily_tool])

    plan = state["plan"]
    feedback = state.get("critic_feedback", "")
    feedback_note = f"\n\nCritic feedback to address: {feedback}" if feedback else ""

    messages = [
        HumanMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Research Plan:\n{plan}{feedback_note}\n\nBegin searching now."),
    ]

    # Agentic loop: keep calling tools until the LLM stops
    findings_parts = []
    for _ in range(8):  # max 8 tool call rounds
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            findings_parts.append(response.content)
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            result = tavily_tool.invoke(tool_call["args"])
            findings_parts.append(str(result))
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    findings = "\n\n".join(findings_parts)
    return {
        **state,
        "research_findings": findings,
        "messages": state["messages"] + [AIMessage(content=f"**Research Findings:**\n{findings}", name="researcher")],
    }
