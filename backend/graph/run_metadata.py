from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.graph.state import AgentState
else:
    AgentState = dict[str, Any]

URL_PATTERN = re.compile(r"https?://[^\s\]'\")},>]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_run_metadata() -> dict[str, Any]:
    return {
        "started_at": utc_now_iso(),
        "completed_at": "",
        "search_rounds": 0,
        "llm_nodes_completed": 0,
    }


def extract_sources(text: str, limit: int = 12) -> list[dict[str, str]]:
    """Extract a compact, de-duplicated source list from search output text."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        domain = url.split("//", 1)[-1].split("/", 1)[0]
        sources.append({"title": domain, "url": url})
        if len(sources) >= limit:
            break

    return sources


def append_warning(state: AgentState, warning: str) -> list[str]:
    warnings = list(state.get("warnings", []))
    if warning and warning not in warnings:
        warnings.append(warning)
    return warnings


def mark_node_complete(state: AgentState, **updates: Any) -> dict[str, Any]:
    metrics = dict(state.get("run_metrics", {}))
    metrics["llm_nodes_completed"] = int(metrics.get("llm_nodes_completed", 0)) + 1
    metrics.update(updates)
    return metrics


def calculate_quality_score(state: AgentState, final_report: str) -> float:
    """Return a simple 0-1 quality signal for UI triage, not a factual guarantee."""
    score = 0.2

    source_count = len(state.get("sources", []))
    score += min(source_count, 6) * 0.08

    if state.get("analysis_result"):
        score += 0.12
    if state.get("chart_image"):
        score += 0.08
    if state.get("critic_verdict") == "pass":
        score += 0.18
    if len(final_report) >= 1200:
        score += 0.08

    revision_count = int(state.get("revision_count", 0))
    if revision_count > 1:
        score -= min(revision_count - 1, 3) * 0.04

    return round(max(0.0, min(score, 1.0)), 2)


def complete_run_metrics(state: AgentState) -> dict[str, Any]:
    metrics = dict(state.get("run_metrics", {}))
    metrics["completed_at"] = utc_now_iso()
    metrics["revision_count"] = state.get("revision_count", 0)
    metrics["source_count"] = len(state.get("sources", []))
    metrics["has_chart"] = bool(state.get("chart_image"))
    return metrics
