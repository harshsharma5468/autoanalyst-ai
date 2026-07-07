import json
import uuid
import logging
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from backend.graph.run_metadata import initial_run_metadata

# AutoAnalyst router — imported eagerly (no DB dependency)
from backend.autoanalyst.router import router as autoanalyst_router

logger = logging.getLogger(__name__)

app = FastAPI(title="AutoAnalyst AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register AutoAnalyst endpoints
app.include_router(autoanalyst_router)

# ── Lazy-load the LangGraph research graph ────────────────────────────────────
# The graph import triggers a PostgreSQL connection attempt via PostgresSaver.
# We defer it to first use so a missing/unreachable DB doesn't prevent the
# server from starting at all (AutoAnalyst endpoints work without it).
_graph = None
_graph_error: str | None = None


def _get_graph():
    global _graph, _graph_error
    if _graph is not None:
        return _graph
    if _graph_error:
        raise RuntimeError(_graph_error)
    try:
        from backend.graph.workflow import build_graph
        _graph = build_graph()
        return _graph
    except Exception as exc:
        _graph_error = str(exc)
        logger.error("Failed to load research graph: %s", exc)
        raise RuntimeError(f"Research graph unavailable: {exc}") from exc


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    session_id: str | None = None
    research_depth: Literal["quick", "standard", "deep"] = "standard"
    report_style: Literal["executive", "technical", "bullet"] = "executive"
    include_chart: bool = True
    max_sources: int = Field(default=8, ge=3, le=15)


@app.get("/health")
def health():
    """Quick liveness check — does NOT require the research graph."""
    return {
        "status": "ok",
        "graph_loaded": _graph is not None,
        "version": app.version,
    }


def _initial_state(query: str, request: QueryRequest):
    return {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "plan": "",
        "research_findings": "",
        "analysis_result": "",
        "critic_verdict": "",
        "critic_feedback": "",
        "revision_count": 0,
        "chart_image": "",
        "sources": [],
        "run_metrics": initial_run_metadata(),
        "warnings": [],
        "quality_score": 0.0,
        "user_preferences": {
            "research_depth": request.research_depth,
            "report_style": request.report_style,
            "include_chart": request.include_chart,
            "max_sources": request.max_sources,
        },
    }


@app.post("/analyze")
def analyze(request: QueryRequest):
    """Non-streaming endpoint — returns the full final report."""
    try:
        graph = _get_graph()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    query = request.query.strip()
    initial_state = _initial_state(query, request)

    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    final_report = next(
        (
            m.content
            for m in reversed(final_state["messages"])
            if getattr(m, "name", "") == "compiler"
        ),
        "No report generated.",
    )

    return {
        "session_id": session_id,
        "report": final_report,
        "chart_image": final_state.get("chart_image", ""),
        "revision_count": final_state.get("revision_count", 0),
        "critic_verdict": final_state.get("critic_verdict", ""),
        "quality_score": final_state.get("quality_score", 0.0),
        "sources": final_state.get("sources", []),
        "run_metrics": final_state.get("run_metrics", {}),
        "warnings": final_state.get("warnings", []),
    }


@app.post("/analyze/stream")
def analyze_stream(request: QueryRequest):
    """SSE streaming endpoint — streams each agent's output as it happens."""
    try:
        graph = _get_graph()
    except RuntimeError as e:
        def _err_gen():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    query = request.query.strip()
    initial_state = _initial_state(query, request)

    def event_generator():
        try:
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        payload = {
                            "session_id": session_id,
                            "node": node_name,
                            "content": msg.content,
                            "chart_image": node_output.get("chart_image", ""),
                            "critic_verdict": node_output.get("critic_verdict", ""),
                            "quality_score": node_output.get("quality_score", 0.0),
                            "sources": node_output.get("sources", []),
                            "run_metrics": node_output.get("run_metrics", {}),
                            "warnings": node_output.get("warnings", []),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
