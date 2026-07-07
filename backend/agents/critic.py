from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from backend.config import get_llm
from backend.graph.state import AgentState
from backend.graph.run_metadata import mark_node_complete

CRITIC_PROMPT = """You are a Critic Agent. Your job is to evaluate whether the research and analysis fully satisfies the user's original query.

Evaluate strictly:
- Are all questions in the original query answered?
- Are specific numbers/metrics present (not vague estimates)?
- Are calculations (e.g., CAGR) actually performed and shown?
- Is the information coherent and non-contradictory?

Respond ONLY with valid JSON in this exact format:
{{
  "verdict": "pass" or "fail",
  "feedback": "Specific explanation of what is missing or wrong. Empty string if pass."
}}
"""


def critic_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0)
    parser = JsonOutputParser()

    response = llm.invoke([
        HumanMessage(content=CRITIC_PROMPT),
        HumanMessage(content=(
            f"Original Query:\n{state['original_query']}\n\n"
            f"Research Findings:\n{state['research_findings']}\n\n"
            f"Analysis Result:\n{state['analysis_result']}"
        )),
    ])

    try:
        result = parser.parse(response.content)
        verdict = result.get("verdict", "fail")
        feedback = result.get("feedback", "")
    except Exception:
        verdict = "fail"
        feedback = f"Critic could not parse its own output. Raw: {response.content}"

    revision_count = state.get("revision_count", 0)

    return {
        **state,
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "revision_count": revision_count + 1,
        "run_metrics": mark_node_complete(state),
        "messages": state["messages"] + [
            AIMessage(
                content=f"**Critic Verdict:** {verdict.upper()}\n{feedback}",
                name="critic",
            )
        ],
    }
