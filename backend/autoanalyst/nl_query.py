"""
nl_query.py
===========
Natural-Language Chat Interface for the AutoAnalyst module.

Given a loaded DataFrame and a user question, the NLQueryEngine:
  1. Builds a data context (schema + sample rows + stats).
  2. Sends question + context to the LLM.
  3. If the LLM returns Python code, executes it safely on the DataFrame.
  4. If a plot was generated, captures it as a base64 PNG.
  5. Returns a structured response dict.

The engine maintains a conversation history so follow-up questions work.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import textwrap
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert data analyst assistant.
You have access to a pandas DataFrame called `df`.

When answering questions:
1. If computation or a chart is needed, write Python code inside a code block:
   ```python
   # your code here
   ```
2. The code has access to:
   - `df`          – the DataFrame
   - `pd`          – pandas
   - `np`          – numpy
   - `plt`         – matplotlib.pyplot (already imported, Agg backend)
   - `result`      – a dict you MUST populate with {"answer": "...", "chart": None}
     Set result["chart"] to the base64-encoded PNG if you create a figure, else None.
3. After the code block, write a concise plain-English explanation.
4. If no computation is needed, answer directly without a code block.
5. Keep answers focused and factual. Do not hallucinate data.
6. For anomaly detection, use IQR or z-score on numeric columns.
"""


# ---------------------------------------------------------------------------
# Safe execution sandbox
# ---------------------------------------------------------------------------

def _safe_exec(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute *code* in a restricted namespace with df available.
    Returns {"answer": str, "chart": str|None, "error": str|None}.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result: Dict[str, Any] = {"answer": "", "chart": None}

    namespace = {
        "df":     df.copy(),
        "pd":     pd,
        "np":     np,
        "plt":    plt,
        "result": result,
        "io":     io,
        "base64": base64,
    }

    try:
        exec(compile(code, "<nl_query>", "exec"), namespace)  # noqa: S102

        # If the code created a matplotlib figure but didn't encode it,
        # capture the current figure automatically.
        if result.get("chart") is None and plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            result["chart"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")

        # Allow code to store the answer in result["answer"]
        return {
            "answer": str(result.get("answer", "See plot above.")),
            "chart":  result.get("chart"),
            "error":  None,
        }
    except Exception:
        plt.close("all")
        tb = traceback.format_exc()
        logger.warning("nl_query exec error:\n%s", tb)
        return {
            "answer": "",
            "chart":  None,
            "error":  tb,
        }


def _extract_code(text: str) -> Optional[str]:
    """Extract the first ```python ... ``` block from *text*."""
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches[0].strip() if matches else None


def _strip_code_block(text: str) -> str:
    """Remove code blocks from text, returning the surrounding prose."""
    return re.sub(r"```[\s\S]*?```", "", text).strip()


# ---------------------------------------------------------------------------
# Data context builder
# ---------------------------------------------------------------------------

def _build_context(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Build a compact text description of the DataFrame for the LLM prompt."""
    lines: List[str] = []

    lines.append(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append("")

    # Schema
    lines.append("Column schema:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = int(df[col].isnull().sum())
        null_pct = f"{100 * n_null / len(df):.1f}%" if len(df) else "0%"
        lines.append(f"  - {col}: {dtype}  (nulls: {n_null} / {null_pct})")
    lines.append("")

    # Sample rows
    lines.append(f"First {max_rows} rows:")
    lines.append(df.head(max_rows).to_string(index=False))
    lines.append("")

    # Numeric stats
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append("Numeric summary:")
        lines.append(df[num_cols].describe().round(3).to_string())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NLQueryEngine
# ---------------------------------------------------------------------------

class NLQueryEngine:
    """
    Chat interface for natural-language data exploration.

    Usage
    -----
    engine = NLQueryEngine(df)
    response = engine.query("Show me sales trends by region")
    print(response["text"])
    if response["chart"]:
        # base64 PNG
        ...

    engine.query("What are the anomalies in revenue?")
    engine.clear_history()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("NLQueryEngine requires a pandas DataFrame.")
        self.df = df
        self._history: List[Dict[str, str]] = []   # [{role, content}]
        self._context: str = _build_context(df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the DataFrame.

        Returns
        -------
        dict with keys:
          "text"          – LLM answer (plain English, no code blocks)
          "chart"         – base64 PNG string or None
          "code_executed" – the Python code that was run (or None)
          "exec_error"    – execution traceback string or None
        """
        if not question.strip():
            return {"text": "Please enter a question.", "chart": None, "code_executed": None, "exec_error": None}

        # Build the user turn: prepend context on first message
        if not self._history:
            user_content = f"Data context:\n{self._context}\n\nQuestion: {question}"
        else:
            user_content = f"Question: {question}"

        self._history.append({"role": "user", "content": user_content})

        # Call LLM
        raw_response = self._call_llm()
        self._history.append({"role": "assistant", "content": raw_response})

        # Execute any code the LLM generated
        code = _extract_code(raw_response)
        exec_result: Dict[str, Any] = {"answer": "", "chart": None, "error": None}

        if code:
            exec_result = _safe_exec(code, self.df)

        # Build the prose explanation (strip code block from LLM response)
        prose = _strip_code_block(raw_response)
        if exec_result.get("error"):
            prose += f"\n\n⚠️ Code execution error:\n```\n{exec_result['error']}\n```"
        elif exec_result.get("answer"):
            # Prefer the programmatically computed answer over LLM prose
            prose = exec_result["answer"] + ("\n\n" + prose if prose else "")

        return {
            "text":          prose.strip(),
            "chart":         exec_result.get("chart"),
            "code_executed": code,
            "exec_error":    exec_result.get("error"),
        }

    def clear_history(self) -> None:
        """Reset conversation history (context is preserved)."""
        self._history.clear()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Swap in a new DataFrame and reset context + history."""
        self.df = df
        self._context = _build_context(df)
        self.clear_history()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self) -> str:
        from backend.config import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        llm = get_llm(temperature=0.2)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for turn in self._history[:-1]:   # all previous turns
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        # Current question
        messages.append(HumanMessage(content=self._history[-1]["content"]))

        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("LLM call failed in NLQueryEngine: %s", exc)
            raise RuntimeError(f"NL query failed: {exc}") from exc
