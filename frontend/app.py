"""
AutoAnalyst AI  —  Frontend
============================
Single-file Streamlit application.

Sections
--------
  Sidebar   : Navigation + file picker + session controls
  Page 1    : Upload & Ingest
  Page 2    : Analysis Dashboard  (EDA + Plotly + ML)
  Page 3    : Chat Assistant      (NL query)
  Page 4    : SQL Generator       (NL → PostgreSQL + clipboard copy)
  Legacy    : Research Agent      (original multi-agent chat)

Backend integration
-------------------
All data processing is delegated to the FastAPI backend at API_URL.
The relevant backend modules are:
  • backend/autoanalyst/data_ingestion.py   → POST /autoanalyst/ingest
  • backend/autoanalyst/analysis_engine.py  → POST /autoanalyst/analyze
  • backend/autoanalyst/nl_query.py         → POST /autoanalyst/nl-query
  • backend/autoanalyst/sql_generator.py    → POST /autoanalyst/sql-generate
"""

# ── Standard library ────────────────────────────────────────────────────────
import hashlib
import json
import os
import re
from io import BytesIO

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ── Backend URL ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoAnalyst AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Brand palette ───────────────────────────────────── */
    :root {
        --brand-primary:   #4F8EF7;
        --brand-secondary: #1E293B;
        --brand-accent:    #38BDF8;
        --card-bg:         #F8FAFC;
        --card-border:     #E2E8F0;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--brand-secondary) !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #475569;
        border-radius: 8px;
        color: #CBD5E1 !important;
        font-size: 0.9rem;
        text-align: left;
        width: 100%;
        transition: all 0.15s;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1D4ED8;
        border-color: var(--brand-primary);
        color: #fff !important;
    }
    /* Active nav item — wrap button div in .nav-active to highlight */
    .nav-active > div > button {
        background: var(--brand-primary) !important;
        border-color: var(--brand-primary) !important;
        color: #fff !important;
        font-weight: 600 !important;
    }

    /* ── KPI metrics ─────────────────────────────────────── */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #64748B; }

    /* ── Section headings ────────────────────────────────── */
    .section-heading {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--brand-secondary);
        border-left: 4px solid var(--brand-primary);
        padding-left: 10px;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ── Data summary card ───────────────────────────────── */
    .summary-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
    }

    .research-panel {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.7rem 0 1rem 0;
    }
    .research-chip {
        display: inline-block;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
        color: #075985;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ── Chat bubbles ────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 4px;
    }

    /* ── Hide Streamlit footer ───────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
_defaults: dict = {
    # Research agent (legacy)
    "session_id":      None,
    "history":         [],
    "show_research":   False,
    # AutoAnalyst
    "aa_session_id":   None,
    "aa_files":        {},    # {filename: meta_dict}
    "aa_active_file":  None,
    "aa_eda_cache":    {},    # {filename: eda_result}
    "aa_model_cache":  {},    # {filename: model_result}
    "aa_sql_result":   None,
    "aa_chat_history": [],
    # Navigation — one of: upload | dashboard | chat | sql
    "aa_page":         "upload",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _api_post(endpoint: str, **kwargs):
    """POST to the backend; shows friendly errors and returns None on failure."""
    try:
        r = requests.post(
            f"{API_URL}{endpoint}",
            timeout=kwargs.pop("timeout", 120),
            **kwargs,
        )
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is it running on port 8000?")
    except requests.exceptions.HTTPError as exc:
        st.error(
            f"❌ Backend error {exc.response.status_code}: "
            f"{exc.response.text[:300]}"
        )
    except Exception as exc:
        st.error(f"❌ Request failed: {exc}")
    return None


def _show_b64_image(b64_str: str, caption: str = "") -> None:
    """Decode a base64 PNG and render it with st.image."""
    try:
        img_bytes = base64.b64decode(b64_str)
        st.image(Image.open(BytesIO(img_bytes)), caption=caption, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render image: {exc}")


import base64   # needed by _show_b64_image (also used throughout)

_PLOTLY_CFG = {"displayModeBar": False}


def _render_research_metadata(
    quality_score: float | None,
    sources: list[dict] | None,
    run_metrics: dict | None,
    warnings: list[str] | None,
) -> None:
    """Render compact observability metadata for the multi-agent research run."""
    sources = sources or []
    run_metrics = run_metrics or {}
    warnings = warnings or []

    if quality_score is not None or sources or run_metrics or warnings:
        st.markdown('<p class="section-heading">Research Run Intelligence</p>', unsafe_allow_html=True)

    if quality_score is not None:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Quality signal", f"{quality_score:.0%}")
        q2.metric("Sources", len(sources))
        q3.metric("Revisions", run_metrics.get("revision_count", 0))
        q4.metric("Search rounds", run_metrics.get("search_rounds", 0))

    if warnings:
        with st.expander("Run warnings"):
            for warning in warnings:
                st.warning(warning)

    if sources:
        with st.expander("Sources used"):
            for idx, source in enumerate(sources, start=1):
                title = source.get("title") or source.get("url") or f"Source {idx}"
                url = source.get("url", "")
                if url:
                    st.markdown(f"{idx}. [{title}]({url})")
                else:
                    st.markdown(f"{idx}. {title}")

    if run_metrics:
        with st.expander("Run details"):
            st.json(run_metrics)


def _render_research_control_panel(controls: dict) -> None:
    depth = controls["research_depth"]
    style = controls["report_style"]
    chart = "On" if controls["include_chart"] else "Off"
    max_sources = controls["max_sources"]
    depth_note = {
        "quick": "Fast scan with fewer searches",
        "standard": "Balanced research and analysis",
        "deep": "More search rounds and stronger evidence coverage",
    }.get(depth, "Balanced research and analysis")

    st.markdown(
        f"""
        <div class="research-panel">
            <div style="font-weight:700;color:#1E293B;margin-bottom:0.4rem">
                Advanced research mode
            </div>
            <span class="research-chip">Depth: {depth.title()}</span>
            <span class="research-chip">Style: {style.title()}</span>
            <span class="research-chip">Charts: {chart}</span>
            <span class="research-chip">Sources: up to {max_sources}</span>
            <div style="color:#64748B;font-size:0.9rem;margin-top:0.55rem">
                {depth_note}. These settings are sent to the backend agents with every query.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# ACTIVE DATASET BANNER
# ════════════════════════════════════════════════════════════════════════════

def _active_dataset_banner() -> None:
    """
    Shows a top-of-page banner with the active dataset name, shape, and a
    quick-switch dropdown when multiple datasets are loaded.
    """
    files = st.session_state.aa_files
    active = st.session_state.aa_active_file

    df_choices = [f for f, m in files.items() if m.get("type") == "dataframe"]

    if not df_choices:
        return  # nothing to show

    meta = files.get(active, {})
    shape = meta.get("shape", [])
    cols = meta.get("columns", [])
    shape_str = f"{shape[0]:,} rows × {shape[1]} cols" if shape else "unknown shape"

    banner_col, switch_col = st.columns([3, 2])

    with banner_col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #1E3A5F 0%, #1E293B 100%);
                border: 1px solid #334155;
                border-left: 4px solid #4F8EF7;
                border-radius: 8px;
                padding: 10px 16px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            ">
                <span style="font-size:1.25rem">🗂️</span>
                <div>
                    <span style="color:#94A3B8;font-size:0.7rem;text-transform:uppercase;
                                 letter-spacing:0.07em">Active Dataset</span><br>
                    <span style="color:#E2E8F0;font-weight:700;font-size:1rem">{active}</span>
                    &nbsp;
                    <span style="color:#64748B;font-size:0.8rem">— {shape_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with switch_col:
        if len(df_choices) > 1:
            cur_idx = df_choices.index(active) if active in df_choices else 0
            chosen = st.selectbox(
                "Switch dataset",
                df_choices,
                index=cur_idx,
                key="inline_dataset_switch",
                label_visibility="visible",
            )
            if chosen != active:
                st.session_state.aa_active_file = chosen
                st.session_state.aa_chat_history = []
                st.rerun()
        else:
            total = len(files)
            st.markdown(
                f"<p style='color:#64748B;font-size:0.8rem;padding-top:28px'>"
                f"{total} dataset(s) loaded</p>",
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def _nav_button(label: str, page_key: str) -> None:
    """Sidebar nav button that highlights when its page is active."""
    is_active = st.session_state.aa_page == page_key
    if is_active:
        st.sidebar.markdown('<div class="nav-active">', unsafe_allow_html=True)
    clicked = st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True)
    if is_active:
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state.aa_page = page_key
        st.rerun()


with st.sidebar:
    st.markdown(
        "<h2 style='color:#E2E8F0;margin-bottom:0'>🔬 AutoAnalyst</h2>"
        "<p style='color:#94A3B8;font-size:0.8rem;margin-top:2px'>"
        "AI-powered data analysis</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "<p style='color:#94A3B8;font-size:0.7rem;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:6px'>Navigation</p>",
        unsafe_allow_html=True,
    )
    _nav_button("📁  Upload & Ingest",    "upload")
    _nav_button("📊  Analysis Dashboard", "dashboard")
    _nav_button("💬  Chat Assistant",     "chat")
    _nav_button("🗄️  SQL Generator",      "sql")

    st.markdown("---")

    # Dataset selector (only once files are loaded)
    if st.session_state.aa_files:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.7rem;letter-spacing:0.08em;"
            "text-transform:uppercase;margin-bottom:6px'>Active dataset</p>",
            unsafe_allow_html=True,
        )
        df_choices = [
            f for f, m in st.session_state.aa_files.items()
            if m.get("type") == "dataframe"
        ]
        if df_choices:
            _cur = st.session_state.aa_active_file
            chosen = st.selectbox(
                "dataset",
                df_choices,
                index=df_choices.index(_cur) if _cur in df_choices else 0,
                label_visibility="collapsed",
            )
            if chosen != st.session_state.aa_active_file:
                st.session_state.aa_active_file = chosen
                st.session_state.aa_chat_history = []
                st.rerun()

    st.markdown("---")

    # Research Agent toggle
    if st.button("🧠  Research Agent", key="nav_research", use_container_width=True):
        st.session_state.show_research = not st.session_state.show_research
        st.rerun()

    st.markdown("---")

    # Session info + clear
    if st.session_state.aa_session_id:
        sid_short = st.session_state.aa_session_id[:8]
        st.markdown(
            f"<p style='color:#64748B;font-size:0.72rem'>"
            f"Session <code style='color:#94A3B8'>{sid_short}…</code></p>",
            unsafe_allow_html=True,
        )
    if st.button("🗑️  Clear session", use_container_width=True):
        if st.session_state.aa_session_id:
            try:
                requests.delete(
                    f"{API_URL}/autoanalyst/session/{st.session_state.aa_session_id}",
                    timeout=10,
                )
            except Exception:
                pass
        for k, v in _defaults.items():
            st.session_state[k] = (type(v)() if isinstance(v, (list, dict)) else v)
        st.rerun()



# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & INGEST
# ════════════════════════════════════════════════════════════════════════════

def _render_summary_card(fname: str, meta: dict) -> None:
    """Data Summary Card: KPIs + first-5-rows preview."""
    rows, ncols = meta["shape"]
    col_types = meta.get("col_types", {})

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown(f"**📂 {fname}**")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows",         f"{rows:,}")
    k2.metric("Columns",      ncols)
    k3.metric("Numeric cols", sum(1 for t in col_types.values() if "int" in t or "float" in t))
    k4.metric("Text cols",    sum(1 for t in col_types.values() if "object" in t or "category" in t))

    if meta.get("preview_json"):
        st.markdown("**First 5 rows:**")
        try:
            preview_df = pd.read_json(meta["preview_json"], orient="split")
            st.dataframe(preview_df, hide_index=True, use_container_width=True)
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)


def render_upload_page() -> None:
    st.markdown('<p class="section-heading">📁 Upload & Ingest</p>', unsafe_allow_html=True)
    st.markdown(
        "Upload **structured data** (CSV / Excel), **documents** (PDF / DOCX / TXT), "
        "or **images**. Multiple files are supported."
    )

    # Show active dataset banner if files are already loaded
    if st.session_state.aa_files:
        _active_dataset_banner()
        st.markdown("")

    uploaded = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "txt", "pdf", "docx", "png", "jpg", "jpeg", "webp"],
        help="Structured: CSV/Excel  |  Text: PDF/DOCX/TXT  |  Images: PNG/JPG/WEBP",
        label_visibility="collapsed",
    )

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        ingest_clicked = st.button(
            "⬆️ Ingest Files",
            use_container_width=True,
            disabled=not uploaded,
        )
    with info_col:
        if uploaded:
            st.caption(f"✅ {len(uploaded)} file(s) selected and ready to ingest")
        else:
            st.caption("No files selected yet")

    if ingest_clicked and uploaded:
        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded]
        form_data = {}
        if st.session_state.aa_session_id:
            form_data["session_id"] = st.session_state.aa_session_id

        with st.spinner("Ingesting files — please wait…"):
            resp = _api_post(
                "/autoanalyst/ingest",
                files=files_payload,
                data=form_data,
                timeout=120,
            )

        if resp:
            result = resp.json()
            st.session_state.aa_session_id = result["session_id"]

            for fname, meta in result.get("ingested", {}).items():
                if meta.get("type") == "dataframe":
                    meta.setdefault(
                        "col_types",
                        {c: "object" for c in meta.get("columns", [])},
                    )
                    meta.setdefault("preview_json", None)
                st.session_state.aa_files[fname] = meta

            for fname, err in result.get("errors", {}).items():
                st.error(f"❌ **{fname}**: {err}")

            n_ok = len(result.get("ingested", {}))
            if n_ok:
                st.success(f"✅ {n_ok} file(s) ingested successfully.")
                df_files = [
                    f for f, m in result["ingested"].items()
                    if m.get("type") == "dataframe"
                ]
                if df_files and not st.session_state.aa_active_file:
                    st.session_state.aa_active_file = df_files[0]
            st.rerun()

    # ── Summary cards ─────────────────────────────────────────────────────
    if st.session_state.aa_files:
        st.markdown("---")
        st.markdown(
            '<p class="section-heading">📋 Ingested Files</p>',
            unsafe_allow_html=True,
        )
        for fname, meta in st.session_state.aa_files.items():
            if meta.get("type") == "dataframe":
                _render_summary_card(fname, meta)
            else:
                with st.expander(f"📄 {fname}  —  {meta.get('word_count', '?')} words"):
                    st.text(meta.get("preview", "(no preview available)"))



# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSIS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

# ── Plotly chart factories ───────────────────────────────────────────────────

def _fig_distributions(num_stats: dict) -> go.Figure:
    """Interactive mean ± std bar chart from describe() stats dict."""
    cols   = list(num_stats.keys())
    means  = [num_stats[c].get("mean", 0) for c in cols]
    stds   = [num_stats[c].get("std",  0) for c in cols]
    fig = go.Figure(go.Bar(
        x=cols, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color="#4F8EF7",
        name="Mean ± Std",
    ))
    fig.update_layout(
        title="Mean ± Std Dev per Numeric Column",
        xaxis_title="Column", yaxis_title="Value",
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, margin=dict(t=45, b=50, l=50, r=20),
    )
    return fig


def _fig_heatmap(num_stats: dict) -> go.Figure:
    """
    Normalised outer-product heatmap built from median values.
    (Real Pearson correlations come from the backend PNG; this
     is the interactive Plotly version shown in the tab.)
    """
    cols = list(num_stats.keys())
    medians = [
        num_stats[c].get("50%", 0) or num_stats[c].get("50", 0)
        for c in cols
    ]
    maxv = max(abs(v) for v in medians) or 1
    norm = [v / maxv for v in medians]
    z = [[norm[i] * norm[j] for j in range(len(cols))] for i in range(len(cols))]
    fig = px.imshow(
        z, x=cols, y=cols,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap (normalised)",
    )
    fig.update_layout(
        height=430, margin=dict(t=55, b=40, l=70, r=20),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def _fig_bar_categories(cat_stats: dict) -> go.Figure | None:
    """Bar chart for top values of the first low-cardinality categorical column."""
    if not cat_stats:
        return None
    col  = next(iter(cat_stats))
    top5 = cat_stats[col].get("top_5", {})
    if not top5:
        return None
    labels, values = list(top5.keys()), list(top5.values())
    fig = px.bar(
        x=labels, y=values,
        labels={"x": col, "y": "Count"},
        title=f"Top Values — {col}",
        color=values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        height=330, margin=dict(t=45, b=65, l=45, r=20),
    )
    return fig


def _fig_feature_importance(fi_dict: dict) -> go.Figure:
    """Horizontal bar chart for ML feature importances."""
    items  = sorted(fi_dict.items(), key=lambda x: x[1])
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#4F8EF7",
    ))
    fig.update_layout(
        title="Feature Importances (top 10)",
        xaxis_title="Importance",
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(300, 30 * len(labels) + 80),
        margin=dict(t=45, b=40, l=130, r=20),
    )
    return fig


# ── Main render ──────────────────────────────────────────────────────────────

def render_dashboard_page() -> None:
    st.markdown(
        '<p class="section-heading">📊 Analysis Dashboard</p>',
        unsafe_allow_html=True,
    )

    active = st.session_state.aa_active_file
    if not active:
        st.info("👈 Upload a dataset first, then return here to run the analysis.")
        return

    meta = st.session_state.aa_files.get(active, {})
    if meta.get("type") != "dataframe":
        st.warning("The dashboard is only available for structured (CSV / Excel) data.")
        return

    _active_dataset_banner()

    # ── Control row ──────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 1, 1])
    with cc1:
        run_eda = st.button("🔍 Run Full EDA", use_container_width=True)
    with cc2:
        target_col = st.text_input(
            "target",
            placeholder="Target column for modeling (leave blank to auto-detect)",
            label_visibility="collapsed",
        )
    with cc3:
        run_model = st.button("🤖 Run ML Analysis", use_container_width=True)
    with cc4:
        skip_outliers = st.checkbox("Skip outlier removal", value=False)

    st.markdown("---")

    # ── EDA call ─────────────────────────────────────────────────────────────
    if run_eda:
        with st.spinner("Running EDA pipeline…"):
            resp = _api_post(
                "/autoanalyst/analyze",
                json={
                    "session_id":      st.session_state.aa_session_id,
                    "filename":        active,
                    "skip_modeling":   True,
                    "remove_outliers": not skip_outliers,
                },
                timeout=180,
            )
        if resp:
            st.session_state.aa_eda_cache[active] = resp.json()

    # ── ML call ──────────────────────────────────────────────────────────────
    if run_model:
        with st.spinner("Training model…  (~30 s)"):
            resp = _api_post(
                "/autoanalyst/analyze",
                json={
                    "session_id":      st.session_state.aa_session_id,
                    "filename":        active,
                    "target_col":      target_col.strip() or None,
                    "skip_modeling":   False,
                    "remove_outliers": not skip_outliers,
                },
                timeout=300,
            )
        if resp:
            result = resp.json()
            st.session_state.aa_eda_cache[active]   = result
            st.session_state.aa_model_cache[active] = result.get("model", {})

    # ── Render EDA ───────────────────────────────────────────────────────────
    eda_result = st.session_state.aa_eda_cache.get(active)
    if not eda_result:
        st.markdown(
            "<div style='text-align:center;padding:3rem;color:#94A3B8'>"
            "Click <b>Run Full EDA</b> to generate the report.</div>",
            unsafe_allow_html=True,
        )
        return

    # Smart Clean KPIs
    cs = eda_result.get("clean_summary", {})
    st.markdown('<p class="section-heading">🧹 Smart Clean</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Duplicates removed", cs.get("duplicates_removed", 0))
    m2.metric("Outliers removed",   cs.get("outliers_removed",  0))
    shape = cs.get("final_shape", [])
    m3.metric("Clean shape", f"{shape[0]:,} × {shape[1]}" if shape else "—")
    if cs.get("missing_values"):
        with st.expander("ℹ️ Missing values filled"):
            st.json(cs["missing_values"])

    eda_data  = eda_result.get("eda", {})
    stats     = eda_data.get("stats", {})
    plots_b64 = eda_data.get("plots", {})
    num_stats = stats.get("numeric", {})
    cat_stats = stats.get("categorical", {})

    # Statistical summary table
    if num_stats:
        st.markdown(
            '<p class="section-heading">📈 Statistical Summary</p>',
            unsafe_allow_html=True,
        )
        stats_df = pd.DataFrame(num_stats).T.round(3)
        try:
            st.dataframe(
                stats_df.style.background_gradient(axis=0, cmap="Blues"),
                use_container_width=True,
            )
        except ImportError:
            st.dataframe(stats_df, use_container_width=True)

    # Plotly tabs
    if num_stats:
        st.markdown(
            '<p class="section-heading">📊 Visualisations</p>',
            unsafe_allow_html=True,
        )
        tab_dist, tab_heat, tab_cat = st.tabs(
            ["📊 Distributions", "🌡️ Correlation Heatmap", "🏷️ Categories"]
        )

        with tab_dist:
            st.plotly_chart(
                _fig_distributions(num_stats),
                use_container_width=True,
                config=_PLOTLY_CFG,
            )
            if plots_b64.get("histograms"):
                with st.expander("View detailed per-column histograms"):
                    _show_b64_image(plots_b64["histograms"])

        with tab_heat:
            st.plotly_chart(
                _fig_heatmap(num_stats),
                use_container_width=True,
                config=_PLOTLY_CFG,
            )
            if plots_b64.get("correlation_heatmap"):
                with st.expander("View full correlation matrix"):
                    _show_b64_image(plots_b64["correlation_heatmap"])

        with tab_cat:
            fig_cat = _fig_bar_categories(cat_stats)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True, config=_PLOTLY_CFG)
            else:
                st.info("No low-cardinality categorical columns found.")
            if plots_b64.get("bar_top_category"):
                with st.expander("View detailed bar chart"):
                    _show_b64_image(plots_b64["bar_top_category"])

    # Scatter + time-series side by side
    sc_col, ts_col = st.columns(2)
    with sc_col:
        if plots_b64.get("scatter_top_corr"):
            st.markdown("**Scatter — most correlated pair**")
            _show_b64_image(plots_b64["scatter_top_corr"])
    with ts_col:
        if plots_b64.get("time_series"):
            st.markdown("**Time-series trend**")
            _show_b64_image(plots_b64["time_series"])

    # ML model results
    model_info = st.session_state.aa_model_cache.get(active)
    if model_info:
        st.markdown("---")
        st.markdown(
            '<p class="section-heading">🤖 ML Model Results</p>',
            unsafe_allow_html=True,
        )
        if model_info.get("error"):
            st.warning(model_info["error"])
        else:
            mc1, mc2 = st.columns([1, 2])
            with mc1:
                st.markdown(f"**Model:** `{model_info.get('model_type', 'n/a')}`")
                st.markdown(f"**Target:** `{model_info.get('target', 'n/a')}`")
                for k, v in model_info.get("metrics", {}).items():
                    st.metric(k, round(v, 4))
            with mc2:
                fi = model_info.get("feature_importances")
                if fi:
                    st.plotly_chart(
                        _fig_feature_importance(fi),
                        use_container_width=True,
                        config=_PLOTLY_CFG,
                    )
            if model_info.get("classification_report"):
                with st.expander("📋 Full classification report"):
                    st.text(model_info["classification_report"])
            if model_info.get("forecast_plot"):
                st.markdown("**ARIMA Forecast:**")
                _show_b64_image(model_info["forecast_plot"])



# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHAT ASSISTANT
# ════════════════════════════════════════════════════════════════════════════

def render_chat_page() -> None:
    st.markdown(
        '<p class="section-heading">💬 Chat Assistant</p>',
        unsafe_allow_html=True,
    )

    active = st.session_state.aa_active_file
    if not active:
        st.info("👈 Upload a dataset first, then chat about it here.")
        return

    meta  = st.session_state.aa_files.get(active, {})
    is_df = meta.get("type") == "dataframe"

    if not is_df:
        st.warning(
            "The Chat Assistant works with **structured data** (CSV / Excel). "
            "Select a DataFrame in the sidebar."
        )
        return

    _active_dataset_banner()

    shape = meta.get("shape", [])
    cols  = meta.get("columns", [])
    st.caption(
        f"Columns: `{'`, `'.join(str(c) for c in cols[:6])}`"
        f"{'…' if len(cols) > 6 else ''}"
    )

    # Example questions
    with st.expander("💡 Example questions — click to use"):
        _examples = [
            "What are the top 5 values in the first column?",
            "Show me the distribution of all numeric columns.",
            "Are there any anomalies or outliers in the data?",
            "What is the correlation between the two highest-variance columns?",
            "Show me the trend over time.",
        ]
        for _ex in _examples:
            if st.button(_ex, key=f"chat_ex_{_ex[:24]}"):
                st.session_state._pending_chat = _ex
                st.rerun()

    st.markdown("---")

    # Render chat history
    for msg in st.session_state.aa_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                try:
                    _show_b64_image(msg["chart"])
                except Exception:
                    pass
            if msg.get("code"):
                with st.expander("🐍 Code executed by the assistant"):
                    st.code(msg["code"], language="python")
            if msg.get("exec_error"):
                with st.expander("⚠️ Execution warning"):
                    st.code(msg["exec_error"], language="text")

    # Consume any pending example question
    pending = st.session_state.pop("_pending_chat", None)

    # Chat input
    user_input = st.chat_input(
        "Ask a question about your data…",
        key="aa_chat_input",
    )
    question = pending or user_input

    if question:
        st.session_state.aa_chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                resp = _api_post(
                    "/autoanalyst/nl-query",
                    json={
                        "session_id": st.session_state.aa_session_id,
                        "filename":   active,
                        "question":   question,
                    },
                    timeout=120,
                )

            if resp is None:
                answer, chart_b64, code, err = (
                    "Sorry — I couldn't reach the backend.", None, None, None
                )
            else:
                _d        = resp.json()
                answer    = _d.get("answer", "")
                chart_b64 = _d.get("chart_b64")
                code      = _d.get("code_executed")
                err       = _d.get("exec_error")

            st.markdown(answer or "_(no text response)_")

            if chart_b64:
                try:
                    _show_b64_image(chart_b64)
                except Exception as e:
                    st.warning(f"Could not render chart: {e}")

            if code:
                with st.expander("🐍 Code executed by the assistant"):
                    st.code(code, language="python")
            if err:
                with st.expander("⚠️ Execution warning"):
                    st.code(err, language="text")

        st.session_state.aa_chat_history.append({
            "role":       "assistant",
            "content":    answer,
            "chart":      chart_b64,
            "code":       code,
            "exec_error": err,
        })

    # Footer controls
    if st.session_state.aa_chat_history:
        st.markdown("---")
        fc1, fc2 = st.columns([1, 4])
        with fc1:
            if st.button("🗑️ Clear history", use_container_width=True):
                try:
                    _api_post(
                        "/autoanalyst/clear-history",
                        json={
                            "session_id": st.session_state.aa_session_id,
                            "filename":   active,
                        },
                        timeout=10,
                    )
                except Exception:
                    pass
                st.session_state.aa_chat_history = []
                st.rerun()
        with fc2:
            n_turns = len(st.session_state.aa_chat_history) // 2
            st.caption(f"{n_turns} exchange(s) in history")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SQL GENERATOR
# ════════════════════════════════════════════════════════════════════════════

# JavaScript + HTML for the clipboard copy button
_COPY_SCRIPT = """
<script>
function copySQL(codeId, btnId) {
    const el = document.getElementById(codeId);
    const btn = document.getElementById(btnId);
    if (!el) return;
    const text = el.innerText || el.textContent;
    const succeed = () => {
        if (btn) {
            btn.innerText = '✅ Copied!';
            setTimeout(() => { btn.innerText = '📋 Copy SQL'; }, 1800);
        }
    };
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(succeed).catch(() => fallback(text, succeed));
    } else {
        fallback(text, succeed);
    }
}
function fallback(text, cb) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); cb(); } catch(_) {}
    document.body.removeChild(ta);
}
</script>
"""


def _html_copy_button(sql: str) -> None:
    """Render a dark code block + animated Copy SQL button via components.html."""
    uid     = hashlib.md5(sql.encode()).hexdigest()[:10]
    code_id = f"sqlcode_{uid}"
    btn_id  = f"copybtn_{uid}"

    escaped = sql.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    n_lines = sql.count("\n")
    height  = max(200, n_lines * 22 + 130)

    html = f"""
{_COPY_SCRIPT}
<pre id="{code_id}" style="
    background:#0F172A;color:#E2E8F0;
    border-radius:10px;padding:1rem 1.2rem;
    font-size:0.83rem;overflow-x:auto;
    font-family:'JetBrains Mono','Fira Code',monospace;
    white-space:pre-wrap;line-height:1.5;
">{escaped}</pre>
<button id="{btn_id}"
    onclick="copySQL('{code_id}', '{btn_id}')"
    style="
        display:inline-flex;align-items:center;gap:6px;
        background:#4F8EF7;color:#fff;border:none;
        border-radius:6px;padding:7px 16px;
        font-size:0.83rem;cursor:pointer;margin-top:6px;
        transition:background 0.15s;
    "
    onmouseover="this.style.background='#1D4ED8'"
    onmouseout="this.style.background='#4F8EF7'"
>📋 Copy SQL</button>
"""
    components.html(html, height=height, scrolling=False)


def render_sql_page() -> None:
    st.markdown(
        '<p class="section-heading">🗄️ SQL Generator</p>',
        unsafe_allow_html=True,
    )

    active = st.session_state.aa_active_file
    if not active:
        st.info("👈 Upload a dataset first to enable SQL generation.")
        return

    meta = st.session_state.aa_files.get(active, {})
    if meta.get("type") != "dataframe":
        st.warning("SQL generation is only available for structured (CSV / Excel) data.")
        return

    _active_dataset_banner()

    cols = meta.get("columns", [])

    st.markdown(
        "Describe the query you want in plain English. "
        "The AI generates optimised **PostgreSQL** code from your dataset schema. "
        "Copy it and run it in your own SQL Workbench — "
        "**no data is sent to the database here**."
    )

    # Schema reference
    with st.expander("📋 Schema — available columns"):
        if cols:
            chunks = [cols[i : i + 4] for i in range(0, len(cols), 4)]
            for chunk in chunks:
                sub = st.columns(len(chunk))
                for sc, c in zip(sub, chunk):
                    sc.code(c, language="text")
        else:
            st.info("No column info available yet.")

    # Example prompts
    _sql_examples = [
        "Total revenue per region, sorted highest first",
        "Monthly order count grouped by status",
        "Customers with more than 5 orders in the last 90 days",
        "Top 10 products by average sale price (min 50 transactions)",
        "Rows where sale amount is > 3 std devs from the mean",
    ]
    with st.expander("💡 Example questions — click to use"):
        for _ex in _sql_examples:
            if st.button(_ex, key=f"sqlex_{_ex[:24]}"):
                st.session_state._pending_sql = _ex
                st.rerun()

    pending_q = st.session_state.pop("_pending_sql", "")

    # Input form
    with st.form("sql_form"):
        question = st.text_area(
            "query",
            value=pending_q,
            height=100,
            placeholder="e.g. Show me the top 10 customers by total purchase value in 2024",
            label_visibility="collapsed",
        )
        f1, f2 = st.columns([3, 1])
        with f1:
            submitted = st.form_submit_button("⚡ Generate SQL", use_container_width=True)
        with f2:
            table_name = st.text_input(
                "table", value="data", label_visibility="collapsed"
            )

    if submitted:
        if not question.strip():
            st.warning("Please describe the query you need.")
        else:
            with st.spinner("Generating SQL…"):
                resp = _api_post(
                    "/autoanalyst/sql-generate",
                    json={
                        "session_id": st.session_state.aa_session_id,
                        "filename":   active,
                        "question":   question.strip(),
                        "table_name": table_name.strip() or "data",
                    },
                    timeout=60,
                )
            if resp:
                st.session_state.aa_sql_result = resp.json()
                st.rerun()

    # Render generated SQL
    result = st.session_state.aa_sql_result
    if result:
        raw_sql = result.get("raw_sql", "")
        st.markdown("---")
        st.markdown(
            '<p class="section-heading">Generated SQL</p>',
            unsafe_allow_html=True,
        )
        # Native syntax-highlighted block (for accessibility + fallback)
        st.code(raw_sql, language="sql")
        # Dark-themed block + clipboard copy button
        _html_copy_button(raw_sql)

        if result.get("schema"):
            with st.expander("📋 Schema sent to the AI"):
                st.code(result["schema"], language="sql")

        if st.button("🔄 Generate another query"):
            st.session_state.aa_sql_result = None
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER — dispatch to the active page
# ════════════════════════════════════════════════════════════════════════════

_page = st.session_state.aa_page

if not st.session_state.show_research:
    # Header banner
    st.markdown(
        "<h1 style='margin-bottom:2px'>🔬 AutoAnalyst AI</h1>"
        "<p style='color:#64748B;margin-top:0'>Autonomous data analysis · EDA · ML · SQL generation</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if _page == "upload":
        render_upload_page()
    elif _page == "dashboard":
        render_dashboard_page()
    elif _page == "chat":
        render_chat_page()
    elif _page == "sql":
        render_sql_page()

else:
    # ════════════════════════════════════════════════════════
    # LEGACY — Research Agent (original multi-agent chat)
    # ════════════════════════════════════════════════════════
    import json as _json

    st.markdown(
        "<h1 style='margin-bottom:2px'>🧠 Research Agent</h1>"
        "<p style='color:#64748B;margin-top:0'>Multi-agent web research & analysis</p>",
        unsafe_allow_html=True,
    )

    _stream = st.sidebar.toggle("Stream agent thoughts", value=True)
    st.sidebar.markdown("### Research controls")
    _depth_label = st.sidebar.selectbox(
        "Depth",
        options=["Quick", "Standard", "Deep"],
        index=1,
    )
    _style_label = st.sidebar.selectbox(
        "Report style",
        options=["Executive", "Technical", "Bullet"],
        index=0,
    )
    _include_chart = st.sidebar.toggle("Generate chart", value=True)
    _max_sources = st.sidebar.slider("Max sources", min_value=3, max_value=15, value=8)
    _research_controls = {
        "research_depth": _depth_label.lower(),
        "report_style": _style_label.lower(),
        "include_chart": _include_chart,
        "max_sources": _max_sources,
    }
    _render_research_control_panel(_research_controls)

    _examples = [
        "Analyze the latest AI chip market landscape and estimate Nvidia's market share trend.",
        "Compare top EV adoption countries and project growth to 2030 with a chart.",
        "Research solid-state battery bottlenecks, key players, and expected CAGR.",
    ]
    with st.expander("Example research prompts"):
        for _idx, _example in enumerate(_examples):
            if st.button(_example, key=f"research_example_{_idx}", use_container_width=True):
                st.session_state._pending_research_query = _example
                st.rerun()

    for _entry in st.session_state.history:
        with st.chat_message(_entry["role"]):
            st.markdown(_entry["content"])
            if any(_entry.get(k) for k in ("quality_score", "sources", "run_metrics", "warnings")):
                _render_research_metadata(
                    _entry.get("quality_score"),
                    _entry.get("sources", []),
                    _entry.get("run_metrics", {}),
                    _entry.get("warnings", []),
                )
            if _entry.get("chart"):
                try:
                    _show_b64_image(_entry["chart"])
                except Exception:
                    pass

    _pending_query = st.session_state.pop("_pending_research_query", None)
    _query = _pending_query or st.chat_input("Ask a complex research question…")

    if _query:
        st.session_state.history.append({"role": "user", "content": _query})
        with st.chat_message("user"):
            st.markdown(_query)

        with st.chat_message("assistant"):
            if _stream:
                _thought_box  = st.container()
                _final_holder = st.empty()
                _chart_holder = st.empty()
                _node_outputs: dict = {}
                _final_report = ""
                _chart_b64    = ""
                _quality_score = None
                _sources       = []
                _run_metrics   = {}
                _warnings      = []

                with st.spinner("Agents working…"):
                    try:
                        with requests.post(
                            f"{API_URL}/analyze/stream",
                            json={
                                "query":      _query,
                                "session_id": st.session_state.session_id,
                                **_research_controls,
                            },
                            stream=True,
                            timeout=300,
                        ) as _resp:
                            for _line in _resp.iter_lines():
                                if not _line:
                                    continue
                                _line = _line.decode("utf-8")
                                if not _line.startswith("data:"):
                                    continue
                                _ds = _line[5:].strip()
                                if _ds == "[DONE]":
                                    break
                                try:
                                    _d = _json.loads(_ds)
                                except _json.JSONDecodeError:
                                    continue
                                if "error" in _d:
                                    st.error(_d["error"])
                                    break
                                if _d.get("session_id"):
                                    st.session_state.session_id = _d["session_id"]
                                _node    = _d.get("node", "unknown")
                                _content = _d.get("content", "")
                                if _d.get("chart_image"):
                                    _chart_b64 = _d["chart_image"]
                                if _d.get("quality_score"):
                                    _quality_score = float(_d["quality_score"])
                                if _d.get("sources"):
                                    _sources = _d["sources"]
                                if _d.get("run_metrics"):
                                    _run_metrics = _d["run_metrics"]
                                if _d.get("warnings"):
                                    _warnings = _d["warnings"]
                                _node_outputs.setdefault(_node, []).append(_content)
                                with _thought_box:
                                    for _n, _parts in _node_outputs.items():
                                        _icon = {
                                            "supervisor": "🧠", "researcher": "🔍",
                                            "analyst": "📊", "critic": "⚖️",
                                            "compiler": "📝",
                                        }.get(_n, "🤖")
                                        with st.expander(
                                            f"{_icon} {_n.capitalize()} Agent",
                                            expanded=(_n == _node),
                                        ):
                                            st.markdown("\n\n".join(_parts))
                                if _node == "compiler":
                                    _final_report = _content
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")

                if _final_report:
                    _final_holder.markdown(_final_report)
                    _render_research_metadata(_quality_score, _sources, _run_metrics, _warnings)
                if _chart_b64:
                    try:
                        _show_b64_image(_chart_b64, caption="Generated Chart")
                    except Exception:
                        pass

            else:
                with st.spinner("Analysing…"):
                    try:
                        _r = requests.post(
                            f"{API_URL}/analyze",
                            json={
                                "query":      _query,
                                "session_id": st.session_state.session_id,
                                **_research_controls,
                            },
                            timeout=300,
                        )
                        _r.raise_for_status()
                        _d = _r.json()
                        st.session_state.session_id = _d["session_id"]
                        _final_report = _d["report"]
                        _chart_b64    = _d.get("chart_image", "")
                        _quality_score = _d.get("quality_score")
                        _sources       = _d.get("sources", [])
                        _run_metrics   = _d.get("run_metrics", {})
                        _warnings      = _d.get("warnings", [])
                        st.markdown(_final_report)
                        _render_research_metadata(_quality_score, _sources, _run_metrics, _warnings)
                        if _chart_b64:
                            _show_b64_image(_chart_b64, caption="Generated Chart")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")
                        _final_report = ""

            if _final_report:
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": _final_report,
                    "chart":   _chart_b64,
                    "quality_score": _quality_score,
                    "sources": _sources,
                    "run_metrics": _run_metrics,
                    "warnings": _warnings,
                })
