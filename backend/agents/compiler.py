from langchain_core.messages import HumanMessage, AIMessage
from backend.config import get_llm
from backend.graph.state import AgentState

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

    response = llm.invoke([
        HumanMessage(content=COMPILER_PROMPT),
        HumanMessage(content=(
            f"Original Query:\n{state['original_query']}\n\n"
            f"Research Findings:\n{state['research_findings']}\n\n"
            f"Analysis Result:\n{state['analysis_result']}"
        )),
    ])

    final_report = response.content
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=final_report, name="compiler")],
    }
