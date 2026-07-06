"""
sql_generator.py
================
Natural-Language → optimised PostgreSQL query generator.

Given:
  - A pandas DataFrame (or its schema string)
  - A plain-English question from the user

Returns:
  - A formatted SQL code block (string)
  - The raw SQL string

The generator uses the existing project LLM (get_llm()) so it respects
the LLM_PROVIDER / LLM_MODEL env vars.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_schema_str(df: pd.DataFrame, table_name: str = "data") -> str:
    """
    Convert a DataFrame schema into a compact SQL-like CREATE TABLE statement
    that the LLM can reason about.
    """
    pg_type_map = {
        "int64":          "INTEGER",
        "int32":          "INTEGER",
        "float64":        "DOUBLE PRECISION",
        "float32":        "REAL",
        "bool":           "BOOLEAN",
        "object":         "TEXT",
        "category":       "TEXT",
        "datetime64[ns]": "TIMESTAMP",
    }

    lines = [f"CREATE TABLE {table_name} ("]
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        # Handle datetime with tz
        if "datetime" in dtype_str:
            pg_type = "TIMESTAMP"
        else:
            pg_type = pg_type_map.get(dtype_str, "TEXT")
        safe_col = f'"{col}"' if " " in col or not col.isidentifier() else col
        lines.append(f"    {safe_col}  {pg_type},")
    # Drop trailing comma on last line
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")

    # Append sample values (first 3 rows) as a comment to help the LLM
    lines.append("")
    lines.append("-- Sample rows:")
    sample = df.head(3)
    for _, row in sample.iterrows():
        row_str = ", ".join(
            f"{col}={repr(val)}" for col, val in row.items()
        )
        lines.append(f"--   {row_str}")

    # Stats summary for numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append("-- Numeric stats (min / max):")
        for col in num_cols[:5]:
            mn = df[col].min()
            mx = df[col].max()
            lines.append(f"--   {col}: min={mn}, max={mx}")

    return "\n".join(lines)


def _extract_sql(raw: str) -> str:
    """
    Extract the first SQL code block from a markdown-formatted LLM response.
    Falls back to returning the raw string if no code block is found.
    """
    # Match ```sql ... ``` or ``` ... ```
    pattern = r"```(?:sql)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, raw, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    # Try to find something that looks like SELECT/WITH
    sql_start = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", raw, re.IGNORECASE)
    if sql_start:
        return raw[sql_start.start():].strip()
    return raw.strip()


def _wrap_code_block(sql: str) -> str:
    return f"```sql\n{sql}\n```"


# ---------------------------------------------------------------------------
# SQLGenerator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert PostgreSQL data analyst.
Given a table schema and a natural-language question, write a single, \
optimised PostgreSQL query that answers the question.

Rules:
1. Output ONLY the SQL query inside a markdown code block: ```sql ... ```
2. Use proper PostgreSQL syntax (e.g. ILIKE for case-insensitive text match, \
DATE_TRUNC for time grouping, ROUND(...::numeric, 2) for rounding).
3. Add brief inline SQL comments (-- ...) to explain non-obvious parts.
4. Never use SELECT *; always name the columns explicitly.
5. Include LIMIT 1000 unless the question explicitly asks for aggregates.
6. If the question is ambiguous, make a reasonable assumption and add a \
SQL comment explaining it.
7. Do NOT include any text outside the code block.
"""


class SQLGenerator:
    """
    Converts natural-language questions into PostgreSQL queries.

    Usage
    -----
    gen = SQLGenerator()
    result = gen.generate(df=df, question="What is total sales per region?")
    print(result["sql_block"])   # formatted markdown code block
    print(result["raw_sql"])     # raw SQL string
    """

    def __init__(self, table_name: str = "data") -> None:
        self.table_name = table_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        schema_str: Optional[str] = None,
    ) -> dict:
        """
        Generate a PostgreSQL query for *question*.

        Parameters
        ----------
        question   : the user's natural-language question.
        df         : a DataFrame whose schema will be used (preferred).
        schema_str : manual schema string, used when df is not available.

        Returns
        -------
        dict with keys:
          "raw_sql"   – the SQL string
          "sql_block" – the SQL wrapped in a ```sql ``` markdown code block
          "schema"    – the schema string sent to the LLM
        """
        if df is None and schema_str is None:
            raise ValueError("Provide either a DataFrame (df) or a schema string (schema_str).")

        schema = schema_str or _build_schema_str(df, self.table_name)

        user_message = (
            f"Table schema:\n\n{schema}\n\n"
            f"Question: {question}"
        )

        raw_response = self._call_llm(user_message)
        raw_sql = _extract_sql(raw_response)
        sql_block = _wrap_code_block(raw_sql)

        return {
            "raw_sql":   raw_sql,
            "sql_block": sql_block,
            "schema":    schema,
        }

    def generate_from_schema_dict(
        self,
        schema_dict: dict,
        question: str,
    ) -> dict:
        """
        Convenience method: pass a {column_name: pg_type} dict instead of a DataFrame.
        """
        lines = [f"CREATE TABLE {self.table_name} ("]
        for col, pg_type in schema_dict.items():
            safe_col = f'"{col}"' if " " in col else col
            lines.append(f"    {safe_col}  {pg_type},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        schema_str = "\n".join(lines)
        return self.generate(question=question, schema_str=schema_str)

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str) -> str:
        from backend.config import get_llm

        try:
            llm = get_llm(temperature=0)
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("LLM call failed in SQLGenerator: %s", exc)
            raise RuntimeError(f"SQL generation failed: {exc}") from exc
