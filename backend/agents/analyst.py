from langchain_core.messages import HumanMessage, AIMessage
from backend.config import get_llm
from backend.tools.code_executor import run_code_in_sandbox
from backend.graph.state import AgentState
from backend.graph.run_metadata import append_warning, mark_node_complete
import re

ANALYST_PROMPT = """You are a Data Analyst Agent. You write and execute Python code to analyze data.

Given research findings and a plan, you must:
1. Write clean Python code to perform the required calculations (e.g., CAGR, growth rates, market share).
2. If a chart is needed, use matplotlib and call plt.savefig('chart.png') at the end.
3. Print all key results clearly with labels.
4. Wrap your code in a ```python ... ``` block.

Available libraries: pandas, numpy, matplotlib, scipy.
Do NOT use external APIs or file I/O other than saving chart.png.
"""


def _extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def analyst_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0)

    plan = state["plan"]
    findings = state["research_findings"]
    preferences = state.get("user_preferences", {})
    feedback = state.get("critic_feedback", "")
    feedback_note = f"\n\nCritic feedback to address: {feedback}" if feedback else ""
    chart_instruction = (
        "Generate a chart when it clarifies the answer."
        if preferences.get("include_chart", True)
        else "Do not generate charts unless absolutely necessary."
    )

    response = llm.invoke([
        HumanMessage(content=ANALYST_PROMPT),
        HumanMessage(content=(
            f"Research Plan:\n{plan}\n\n"
            f"Research Findings:\n{findings}"
            f"{feedback_note}\n\n"
            f"Chart preference: {chart_instruction}\n"
            "Write the Python analysis code now."
        )),
    ])

    code = _extract_code(response.content)
    analysis_result = response.content
    chart_image = ""
    warnings = list(state.get("warnings", []))

    if code:
        execution = run_code_in_sandbox(code)
        output_parts = []
        if execution["stdout"]:
            output_parts.append(f"Output:\n{execution['stdout']}")
        if execution["stderr"]:
            output_parts.append(f"Stderr:\n{execution['stderr']}")
        if execution["error"]:
            output_parts.append(f"Error:\n{execution['error']}")
        analysis_result = response.content + "\n\n" + "\n".join(output_parts)
        chart_image = execution.get("chart_image", "")
        if execution.get("error") or execution.get("stderr"):
            warnings = append_warning(state, "Analyst code executed with warnings or errors.")
    else:
        warnings = append_warning(state, "Analyst did not produce executable Python code.")

    return {
        **state,
        "analysis_result": analysis_result,
        "chart_image": chart_image,
        "warnings": warnings,
        "run_metrics": mark_node_complete(state, generated_code=bool(code)),
        "messages": state["messages"] + [AIMessage(content=f"**Analysis Result:**\n{analysis_result}", name="analyst")],
    }
