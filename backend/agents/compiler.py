from langchain_core.messages import HumanMessage, AIMessage
from backend.config import get_llm
from backend.graph.state import AgentState
from backend.graph.run_metadata import calculate_quality_score, complete_run_metrics, mark_node_complete

COMPILER_PROMPT = """You are a Report Compiler. Synthesize the research findings and analysis results into a single, 
well-structured final report for the user.

Format the report with clear sections:
- Executive Summary
- Key Findings (with data points and sources)
- Quantitative Analysis (calculations, CAGR, etc.)
- Conclusion

Be concise, factual, and professional.
"""


def compiler_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0)
    preferences = state.get("user_preferences", {})
    style = preferences.get("report_style", "executive")
    style_instruction = {
        "executive": "Write for a busy decision-maker: crisp summary, clear implications, no excess detail.",
        "technical": "Write for a technical analyst: include methodology, assumptions, calculations, and caveats.",
        "bullet": "Write in concise bullets with short sections and direct takeaways.",
    }.get(style, "Write a concise professional report.")

    response = llm.invoke([
        HumanMessage(content=COMPILER_PROMPT),
        HumanMessage(content=(
            f"Original Query:\n{state['original_query']}\n\n"
            f"Research Findings:\n{state['research_findings']}\n\n"
            f"Analysis Result:\n{state['analysis_result']}\n\n"
            f"Report Style: {style}\n{style_instruction}"
        )),
    ])

    final_report = response.content
    quality_score = calculate_quality_score(state, final_report)
    run_metrics = mark_node_complete(state)
    run_metrics.update(complete_run_metrics(state))
    return {
        **state,
        "quality_score": quality_score,
        "run_metrics": run_metrics,
        "messages": state["messages"] + [AIMessage(content=final_report, name="compiler")],
    }
