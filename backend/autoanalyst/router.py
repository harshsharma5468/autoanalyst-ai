"""
router.py
=========
FastAPI router for the AutoAnalyst module.

Mounts at  /autoanalyst  (registered in backend/api/main.py).

Endpoints
---------
POST /autoanalyst/ingest          – upload one or more files
POST /autoanalyst/analyze         – run SmartClean + EDA + Modeling on ingested data
POST /autoanalyst/nl-query        – natural-language question → answer + optional chart
POST /autoanalyst/sql-generate    – natural-language question → PostgreSQL query
GET  /autoanalyst/session/{id}    – list files in a session
DELETE /autoanalyst/session/{id}  – clear a session
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.autoanalyst.data_ingestion import FileProcessor
from backend.autoanalyst.analysis_engine import AnalysisEngine
from backend.autoanalyst.sql_generator import SQLGenerator
from backend.autoanalyst.nl_query import NLQueryEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autoanalyst", tags=["AutoAnalyst"])

# ---------------------------------------------------------------------------
# In-memory session store
# (For production, use Redis or a database-backed store)
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}
# Structure per session:
# {
#   "processor": FileProcessor,
#   "nl_engines": {filename: NLQueryEngine},
# }


def _get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "processor": FileProcessor(),
            "nl_engines": {},
        }
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    session_id: str
    filename: str                          # which ingested file to analyse
    target_col: Optional[str] = None       # optional target column for modeling
    remove_outliers: bool = True
    skip_modeling: bool = False


class NLQueryRequest(BaseModel):
    session_id: str
    filename: str
    question: str


class SQLGenerateRequest(BaseModel):
    session_id: str
    filename: str
    question: str
    table_name: str = "data"


class ClearHistoryRequest(BaseModel):
    session_id: str
    filename: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest_files(
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    """
    Upload one or more files for a session.
    If session_id is omitted, a new session UUID is created.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session_id = session_id or str(uuid.uuid4())
    session = _get_or_create_session(session_id)
    processor: FileProcessor = session["processor"]

    results = {}
    errors = {}

    for upload in files:
        filename = upload.filename or f"file_{uuid.uuid4().hex[:6]}"
        data = await upload.read()
        try:
            result = processor.ingest(filename=filename, data=data)
            if isinstance(result, pd.DataFrame):
                results[filename] = {
                    "type": "dataframe",
                    "shape": list(result.shape),
                    "columns": list(result.columns),
                }
            else:
                results[filename] = {
                    "type": "text",
                    "word_count": len(result.split()),
                    "preview": result[:500],
                }
        except Exception as exc:
            logger.error("Ingest error for '%s': %s", filename, exc)
            errors[filename] = str(exc)

    return {
        "session_id": session_id,
        "ingested": results,
        "errors": errors,
        "summary": processor.summary(),
    }


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Run SmartClean + EDA + Modeling on a previously ingested DataFrame.
    Returns statistics and base64-encoded plot images.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    processor: FileProcessor = session["processor"]
    dfs = processor.get_dataframes()

    if request.filename not in dfs:
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found in session or is not structured data.",
        )

    df = dfs[request.filename]

    try:
        engine = AnalysisEngine(
            remove_outliers=request.remove_outliers,
            target_col=request.target_col,
            skip_modeling=request.skip_modeling,
        )
        report = engine.run(df)
    except Exception as exc:
        logger.exception("Analysis error for '%s'", request.filename)
        raise HTTPException(status_code=500, detail=str(exc))

    # Flatten the EDA stats so they are JSON-serialisable
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        if hasattr(obj, "item"):        # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):      # numpy array
            return obj.tolist()
        return obj

    return JSONResponse(content=_make_serialisable(report))


@router.post("/nl-query")
def nl_query(request: NLQueryRequest):
    """
    Ask a natural-language question about an ingested DataFrame.
    Returns a text answer and optionally a base64-encoded chart PNG.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    processor: FileProcessor = session["processor"]
    dfs = processor.get_dataframes()

    if request.filename not in dfs:
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found or is not structured data.",
        )

    df = dfs[request.filename]

    # Reuse or create the NL engine for this file (maintains conversation history)
    nl_engines: Dict[str, NLQueryEngine] = session["nl_engines"]
    if request.filename not in nl_engines:
        nl_engines[request.filename] = NLQueryEngine(df)

    engine = nl_engines[request.filename]

    try:
        result = engine.query(request.question)
    except Exception as exc:
        logger.exception("NL query error")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "answer":         result["text"],
        "chart_b64":      result["chart"],
        "code_executed":  result["code_executed"],
        "exec_error":     result["exec_error"],
    }


@router.post("/sql-generate")
def sql_generate(request: SQLGenerateRequest):
    """
    Generate an optimised PostgreSQL query from a natural-language question
    using the schema of an ingested DataFrame.
    Returns the raw SQL and a formatted ```sql``` code block.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    processor: FileProcessor = session["processor"]
    dfs = processor.get_dataframes()

    if request.filename not in dfs:
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found or is not structured data.",
        )

    df = dfs[request.filename]

    try:
        gen = SQLGenerator(table_name=request.table_name)
        result = gen.generate(question=request.question, df=df)
    except Exception as exc:
        logger.exception("SQL generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "raw_sql":   result["raw_sql"],
        "sql_block": result["sql_block"],
        "schema":    result["schema"],
    }


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """List all files in a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    processor: FileProcessor = session["processor"]
    return {"session_id": session_id, "summary": processor.summary(), "files": processor.list_files()}


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Delete all data for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"session_id": session_id, "status": "cleared"}


@router.post("/clear-history")
def clear_chat_history(request: ClearHistoryRequest):
    """Reset the NL conversation history for a specific file."""
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    nl_engines: Dict[str, NLQueryEngine] = session["nl_engines"]
    if request.filename in nl_engines:
        nl_engines[request.filename].clear_history()
    return {"status": "history_cleared", "filename": request.filename}
