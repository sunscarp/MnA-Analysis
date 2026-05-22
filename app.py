import streamlit as st
from dotenv import load_dotenv
from src.data_fetcher import get_company, Company
from src.valuation import ValuationModel
from src.assumptions import ValuationAssumptions, AssumptionsDashboard
from src.ai_assumptions_ui import render_ai_assumptions_menu
from src.ai_assumptions import AIAssumptionGenerator, apply_ai_assumptions_to_model, AIAssumptionResult
from src.market_utils import MarketDataProvider
import pandas as pd
import numpy as np
import traceback
import json
from dataclasses import asdict
from src.report_generator import MAReportGenerator
import tempfile
import os
from datetime import datetime
from contextlib import contextmanager
from html import escape

load_dotenv()

THEME = {
    "page_bg": "#0A0C10",
    "surface": "#0F1117",
    "surface_raised": "#161B26",
    "border": "#1E2535",
    "accent": "#00B4FF",
    "positive": "#10B981",
    "negative": "#EF4444",
    "warning": "#F59E0B",
    "text": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#475569",
    "positive_soft": "#064E3B",
    "negative_soft": "#450A0A",
    "warning_soft": "#451A03",
    "positive_text": "#6EE7B7",
    "negative_text": "#FECACA",
    "warning_text": "#FDE68A",
}

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg: {THEME['page_bg']};
            --surface: {THEME['surface']};
            --surface-2: {THEME['surface_raised']};
            --border: {THEME['border']};
            --accent: {THEME['accent']};
            --positive: {THEME['positive']};
            --negative: {THEME['negative']};
            --warning: {THEME['warning']};
            --text: {THEME['text']};
            --text-secondary: {THEME['text_secondary']};
            --text-muted: {THEME['text_muted']};
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        .stApp {{ background: var(--bg); color: var(--text); }}

        .main .block-container {{
            max-width: 1440px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 24px;
            padding-right: 24px;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0C1018 0%, #0A0C10 100%);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            padding-top: 0.5rem;
        }}

        .app-shell-title {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 10px;
        }}

        .app-monogram {{
            width: 44px;
            height: 44px;
            border-radius: 999px;
            background: #111827;
            border: 1px solid var(--border);
            color: var(--text);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            letter-spacing: 0.08em;
        }}

        .app-title {{
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: 0.02em;
            margin: 0;
        }}

        .app-subtitle {{
            color: var(--text-secondary);
            font-size: 0.82rem;
            margin-top: 2px;
        }}

        .section-header, .sidebar-section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 12px;
            padding: 12px 14px;
            margin: 16px 0 12px;
        }}

        .section-title {{
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.68rem;
            font-weight: 700;
            margin: 0;
        }}

        .badge-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            background: #101521;
        }}

        .badge-blue {{ color: #C4F1FF; background: rgba(0,180,255,0.12); border-color: rgba(0,180,255,0.35); }}
        .badge-green {{ color: #D1FAE5; background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.35); }}
        .badge-amber {{ color: #FEF3C7; background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.35); }}
        .badge-red {{ color: #FECACA; background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.35); }}

        .metric-grid {{ display: grid; gap: 12px; }}
        .metric-grid.cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .metric-grid.cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        .metric-grid.cols-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
        @media (max-width: 1100px) {{ .metric-grid.cols-4, .metric-grid.cols-3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
        @media (max-width: 760px) {{ .metric-grid.cols-4, .metric-grid.cols-3, .metric-grid.cols-2 {{ grid-template-columns: 1fr; }} }}

        .metric-card {{
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .metric-label {{
            color: var(--text-secondary);
            font-size: 11px;
            line-height: 1.2;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .metric-value {{
            color: var(--text);
            font-size: 28px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            font-feature-settings: 'tnum';
            line-height: 1.05;
        }}
        .metric-delta {{
            display: inline-flex;
            margin-top: 10px;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}
        .metric-delta.positive {{ background: rgba(16,185,129,0.15); color: var(--positive); }}
        .metric-delta.negative {{ background: rgba(239,68,68,0.15); color: var(--negative); }}

        .status-bar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }}
        .status-bar strong {{ color: var(--text); }}

        .banner {{
            border-radius: 12px;
            padding: 14px 16px;
            border-left: 4px solid;
            margin: 10px 0 14px;
        }}
        .banner.success {{ background: {THEME['positive_soft']}; border-color: {THEME['positive']}; color: {THEME['positive_text']}; }}
        .banner.warning {{ background: {THEME['warning_soft']}; border-color: {THEME['warning']}; color: {THEME['warning_text']}; }}
        .banner.error {{ background: {THEME['negative_soft']}; border-color: {THEME['negative']}; color: {THEME['negative_text']}; }}
        .banner.info {{ background: var(--surface-2); border-color: var(--accent); color: var(--text-secondary); }}
        .banner .banner-title {{ font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; font-size: 0.75rem; }}
        .banner .banner-body {{ color: inherit; font-size: 0.92rem; line-height: 1.5; }}

        .loading-strip {{
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(0,180,255,0.08), rgba(0,180,255,0.45), rgba(0,180,255,0.08));
            background-size: 220% 100%;
            animation: shimmer 1.3s linear infinite;
            margin: 10px 0 4px;
        }}
        .loading-message {{ color: var(--text-secondary); font-style: italic; font-size: 0.9rem; }}
        @keyframes shimmer {{ 0% {{ background-position: 0% 0; }} 100% {{ background-position: 220% 0; }} }}

        .deal-summary {{
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin-top: 12px;
        }}
        .deal-summary .summary-title {{
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.68rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .deal-summary .summary-value {{
            color: var(--text);
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            font-feature-settings: 'tnum';
        }}

        .pill-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
        .pill-table th, .pill-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.86rem; }}
        .pill-table th {{ background: var(--surface); color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; text-align: left; }}
        .pill-table td {{ color: var(--text); }}
        .pill-table tbody tr:nth-child(even) td {{ background: var(--surface-2); }}
        .pill-table tbody tr.total-row td {{ background: #1E2535; font-weight: 600; }}
        .pill-table td.numeric {{ text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: 'tnum'; }}
        .pill-table td.positive {{ color: var(--positive); }}
        .pill-table td.negative {{ color: var(--negative); }}
        .pill-table td.warning {{ color: var(--warning); }}

        [data-testid="stMetric"] {{
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; }}
        [data-testid="stMetricValue"] {{ color: var(--text); font-variant-numeric: tabular-nums; font-feature-settings: 'tnum'; }}
        [data-testid="stMetricDelta"] {{ color: var(--text-secondary); }}

        div[data-baseweb="tab-list"] {{
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            border-radius: 12px 12px 0 0;
            gap: 0;
            padding: 0;
            width: 100%;
            overflow: hidden;
        }}
        button[data-baseweb="tab"] {{
            flex: 1 1 0;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            color: var(--text-muted) !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            padding: 14px 16px !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--text) !important;
            border-bottom: 2px solid var(--accent) !important;
        }}
        button[data-baseweb="tab"]:hover {{ color: var(--text-secondary) !important; }}

        div[data-baseweb="input"], div[data-baseweb="textarea"] {{ background: var(--surface-2); border-radius: 8px; }}
        input, textarea {{ background: var(--surface-2) !important; color: var(--text) !important; border-color: var(--border) !important; }}
        input:focus, textarea:focus {{ border-color: var(--accent) !important; box-shadow: none !important; }}
        label {{ color: var(--text-secondary) !important; font-weight: 600 !important; letter-spacing: 0.02em; }}
        [data-baseweb="select"] > div {{ background: var(--surface-2) !important; border-color: var(--border) !important; color: var(--text) !important; }}
        section[data-testid="stSidebar"] div.stButton > button, .stButton > button {{
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            background: #151B27 !important;
            color: var(--text) !important;
            font-weight: 700 !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"], .stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: #00111A !important;
            border-color: var(--accent) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            height: 44px !important;
        }}
        section[data-testid="stSidebar"] div.stButton > button, section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ width: 100% !important; }}

        [data-baseweb="slider"] [role="slider"] {{ background: #FFFFFF !important; border: 2px solid var(--accent) !important; box-shadow: none !important; }}
        [data-baseweb="slider"] div[style*="background-color: rgb(0, 180, 255)"] {{ background-color: var(--accent) !important; }}
        [data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
        hr {{ border-color: var(--border) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@contextmanager
def custom_loading(message: str):
    placeholder = st.empty()
    placeholder.markdown(
        f"<div class='loading-strip'></div><div class='loading-message'>{escape(message)}</div>",
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def format_money(value: float, decimals: int = 1, suffix: str = "B") -> str:
    return f"₹{value:,.{decimals}f}{suffix}"


def format_value(value, kind: str = "text") -> str:
    if value is None:
        return "N/A"
    if kind == "money":
        return f"₹{value:,.2f}"
    if kind == "money_b":
        return f"₹{value/1e9:,.1f}B"
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "multiple":
        return f"{value:.1f}x"
    if kind == "integer":
        return f"{value:,.0f}"
    return str(value)


def render_section_header(title: str, badge: str | None = None):
    badge_html = f"<span class='badge-pill'>{escape(badge)}</span>" if badge else ""
    st.markdown(
        f"""
        <div class='section-header'>
            <div class='section-title'>{escape(title)}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(kind: str, title: str, message: str):
    st.markdown(
        f"""
        <div class='banner {kind}'>
            <div class='banner-title'>{escape(title)}</div>
            <div class='banner-body'>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str | None = None, delta_state: str | None = None):
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_state == "positive" else "negative" if delta_state == "negative" else ""
        delta_html = f"<div class='metric-delta {delta_class}'>{escape(delta)}</div>"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{escape(label)}</div>
            <div class='metric-value'>{escape(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[dict], columns: int = 3):
    cols = st.columns(columns)
    for index, metric in enumerate(metrics):
        with cols[index % columns]:
            render_metric_card(
                metric["label"],
                metric["value"],
                metric.get("delta"),
                metric.get("delta_state"),
            )


def render_table(df: pd.DataFrame, highlight_total: bool = True, conditional_columns: dict[str, str] | None = None):
    df_display = df.copy()
    conditional_columns = conditional_columns or {}
    rows = []
    numeric_columns = df_display.select_dtypes(include=[np.number]).columns.tolist()
    for _, row in df_display.iterrows():
        row_class = ""
        if highlight_total and any(str(value).strip().lower() == "total" for value in row.values):
            row_class = "total-row"
        cells = []
        for column, value in row.items():
            classes = []
            if column in numeric_columns:
                classes.append("numeric")
            if column in conditional_columns:
                try:
                    numeric_value = float(value)
                    if numeric_value > 0:
                        classes.append("positive")
                    elif numeric_value < 0:
                        classes.append("negative")
                except Exception:
                    pass
            if row_class == "total-row":
                classes.append("total-row")
            cells.append(f"<td class='{ ' '.join(classes) }'>{escape(str(value))}</td>")
        rows.append(f"<tr class='{row_class}'>{''.join(cells)}</tr>")

    header_html = "".join(f"<th>{escape(str(col)).upper()}</th>" for col in df_display.columns)
    st.markdown(
        f"""
        <table class='pill-table'>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_status_bar(acquirer: str, target: str, deal_value_b: float, premium_pct: float):
    last_updated = datetime.now().strftime("%H:%M")
    st.markdown(
        f"""
        <div class='status-bar'>
            <strong>{escape(acquirer)}</strong> → <strong>{escape(target)}</strong>
            <span>|</span>
            <span>{format_money(deal_value_b, 0, 'B')} Deal</span>
            <span>|</span>
            <span>{premium_pct:.0f}% Premium</span>
            <span>|</span>
            <span>Last updated: {last_updated}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def invalidate_cached_valuation():
    """Invalidate valuation outputs when interactive assumptions are changed."""
    st.session_state.valuation_results = None
    st.session_state.last_assumptions_hash = None

# Page config
st.set_page_config(
    page_title="M&A Intelligence",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "last_tickers" not in st.session_state:
    st.session_state.last_tickers = ("", "")
if "assumptions" not in st.session_state:
    target_ticker = st.session_state.get("last_tickers", ("", ""))[1]
    st.session_state.assumptions = ValuationAssumptions(ticker=target_ticker)
if "valuation_results" not in st.session_state:
    st.session_state.valuation_results = None
if "last_assumptions_hash" not in st.session_state:
    st.session_state.last_assumptions_hash = None
if "assumptions_source_ticker" not in st.session_state:
    st.session_state.assumptions_source_ticker = None
if "accretion_data" not in st.session_state:
    st.session_state.accretion_data = None  # Store pre-computed accretion data
if "ppa_ppe_write_up_pct" not in st.session_state:
    st.session_state.ppa_ppe_write_up_pct = 10.0
if "ppa_intangibles_write_up_pct" not in st.session_state:
    st.session_state.ppa_intangibles_write_up_pct = 15.0
if "ppa_transaction_fees_pct" not in st.session_state:
    st.session_state.ppa_transaction_fees_pct = 1.0
if "ppa_useful_life_years" not in st.session_state:
    st.session_state.ppa_useful_life_years = 10
if "ppa_interest_rate_pct" not in st.session_state:
    st.session_state.ppa_interest_rate_pct = 9.0
if "ppa_debt_tenure_years" not in st.session_state:
    st.session_state.ppa_debt_tenure_years = 5
if "ai_generated_assumptions" not in st.session_state:
    st.session_state.ai_generated_assumptions = None
if "ai_rationales" not in st.session_state:
    st.session_state.ai_rationales = None
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = None
if "ai_key_risks" not in st.session_state:
    st.session_state.ai_key_risks = None
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "base"
if "ai_assumptions_active" not in st.session_state:
    st.session_state.ai_assumptions_active = False
if "ai_raw_response" not in st.session_state:
    st.session_state.ai_raw_response = ""
if "assumptions_widget_version" not in st.session_state:
    st.session_state.assumptions_widget_version = 0

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .header { font-size: 2.5rem; font-weight: 700; color: #00B4FF; }
    .metric-value { font-size: 1.8rem; font-weight: 600; }
    .warning-box {
        background-color: #332701;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FBBF24;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #064E3B;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

def detect_sector(ticker: str) -> str:
    """Detect sector from ticker or company info"""
    ticker_upper = ticker.upper()
    
    # Indian sector mapping
    if any(x in ticker_upper for x in ["INFY", "TCS", "WIPRO", "TECHM", "HCL"]):
        return "technology"
    elif any(x in ticker_upper for x in ["RELIANCE", "RIL"]):
        return "energy"
    elif any(x in ticker_upper for x in ["INDIGO", "INTERGLOBE"]):
        return "airlines"
    elif any(x in ticker_upper for x in ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]):
        return "financial"
    elif any(x in ticker_upper for x in ["ITC", "HINDUNILVR", "TITAN", "NESTLE"]):
        return "consumer"
    elif any(x in ticker_upper for x in ["L&T", "SIEMENS", "ABB"]):
        return "industrial"
    else:
        return "general"


def render_ai_assumptions_ui(company, acquirer, sector: str):
    """Render AI assumptions generation UI in sidebar or tab."""

    st.markdown("### 🤖 AI Assumptions Generator")
    st.caption("Powered by Groq LLM - Sector-specific intelligent assumptions")

    if company is None or acquirer is None:
        render_banner("info", "Context Required", "Run analysis once to load target and acquirer context for AI assumptions.")
        return

    mode_options = ["conservative", "base", "optimistic", "strategic"]
    current_mode = st.session_state.ai_mode if st.session_state.ai_mode in mode_options else "base"
    mode = st.radio(
        "Assumption Mode",
        mode_options,
        index=mode_options.index(current_mode),
        horizontal=True,
        help="Conservative = lower estimates, Strategic = higher control premium & synergies",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🎯 Generate AI Assumptions", type="primary", use_container_width=True):
            with custom_loading("AI is analyzing sector context and generating assumptions..."):
                try:
                    api_key = os.getenv("GROQ_API_KEY", "").strip()
                    generator = AIAssumptionGenerator(api_key=api_key)

                    result = generator.generate_assumptions(
                        company=company,
                        acquirer=acquirer,
                        sector=sector,
                        user_mode=mode,
                        temperature=0.3,
                    )

                    if result.success:
                        st.session_state.ai_generated_assumptions = result.assumptions
                        st.session_state.ai_rationales = result.rationales
                        st.session_state.ai_confidence = result.confidence
                        st.session_state.ai_key_risks = result.key_risks
                        st.session_state.ai_mode = mode
                        render_banner("success", "AI Generated", f"AI assumptions generated. Confidence: {escape(str(result.confidence))}")
                    else:
                        render_banner("error", "AI Generation Failed", escape(str(result.error)))

                except Exception as e:
                    render_banner("error", "Error", escape(str(e)))
                    render_banner("info", "Setup", "Make sure Groq API key is valid. Install: pip install groq")

    with col2:
        if st.button(
            "📋 Apply to Model",
            use_container_width=True,
            disabled=st.session_state.ai_generated_assumptions is None,
        ):
            if st.session_state.ai_generated_assumptions:
                temp_result = AIAssumptionResult(
                    assumptions=st.session_state.ai_generated_assumptions,
                    rationales=st.session_state.ai_rationales or {},
                    confidence=st.session_state.ai_confidence or "Medium",
                    key_risks=st.session_state.ai_key_risks or [],
                    raw_response="",
                    success=True,
                    error="",
                )
                st.session_state.assumptions = apply_ai_assumptions_to_model(
                    st.session_state.assumptions, temp_result
                )
                st.session_state.valuation_results = None
                st.session_state.last_assumptions_hash = None
                render_banner("success", "Assumptions Applied", "Re-run analysis to see impact.")
                st.rerun()

    if st.session_state.ai_generated_assumptions:
        with st.expander("📊 View AI-Generated Assumptions", expanded=False):
            st.markdown(f"**Mode:** {st.session_state.ai_mode.upper()}")
            st.markdown(f"**Confidence:** {st.session_state.ai_confidence}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### DCF Assumptions")
                dcf = st.session_state.ai_generated_assumptions.get("dcf", {})
                st.write(f"- Stage 1 Growth: {dcf.get('stage1_growth', 0):.1%}")
                st.write(f"- Terminal Growth: {dcf.get('terminal_growth', 0):.1%}")
                st.write(f"- Stage 1 EBITDA Margin: {dcf.get('stage1_ebitda_margin', 0):.1%}")
                st.write(f"- Terminal Margin: {dcf.get('terminal_margin', 0):.1%}")
                st.write(f"- Reinvestment Rate: {dcf.get('reinvestment_rate', 0):.1%}")
                st.write(f"- Target D/E: {dcf.get('target_debt_to_capital', 0):.1%}")

                st.markdown("#### Market Multiples")
                comps = st.session_state.ai_generated_assumptions.get("comps", {})
                st.write(f"- EV/EBITDA: {comps.get('ev_ebitda_multiple', 0):.1f}x")
                st.write(f"- P/E: {comps.get('pe_multiple', 0):.1f}x")

            with col2:
                st.markdown("#### Macro & Deal Terms")
                macro = st.session_state.ai_generated_assumptions.get("macro", {})
                st.write(f"- Risk-Free Rate: {macro.get('risk_free_rate', 0):.1%}")
                st.write(f"- Equity Risk Premium: {macro.get('equity_risk_premium', 0):.1%}")
                st.write(f"- Tax Rate: {macro.get('tax_rate', 0):.1%}")

                precedent = st.session_state.ai_generated_assumptions.get("precedent", {})
                st.write(f"- Control Premium: {precedent.get('control_premium', 0):.1%}")
                st.write(f"- Transaction EV/EBITDA: {precedent.get('transaction_ev_ebitda', 0):.1f}x")

                synergies = st.session_state.ai_generated_assumptions.get("synergies", {})
                st.markdown("#### Synergy Estimates")
                st.write(f"- Revenue Synergy: {synergies.get('revenue_synergy_pct', 0):.1%} of revenue")
                st.write(f"- Cost Synergy: {synergies.get('cost_synergy_pct', 0):.1%} of costs")
                st.write(f"- Ramp-up Years: {synergies.get('ramp_up_years', 2)}")

            with st.expander("💡 AI Rationales", expanded=False):
                rationales = st.session_state.ai_rationales or {}
                for key, rationale in rationales.items():
                    st.markdown(f"**{key}:** {rationale}")

            if st.session_state.ai_key_risks:
                st.markdown("#### ⚠️ Key Risks Identified by AI")
                for risk in st.session_state.ai_key_risks:
                    st.write(f"- {risk}")


def derive_assumptions_from_company(company, assumptions):
    """Update assumptions with company-derived values."""
    metrics = company.get_key_metrics()

    derived_growth = MarketDataProvider.derive_growth_rate(company)
    assumptions.stage1_growth = derived_growth

    margins = MarketDataProvider.derive_margins(company)
    if margins.get("ebitda_margin"):
        assumptions.stage1_ebitda_margin = margins["ebitda_margin"]
        assumptions.terminal_margin = max(margins["ebitda_margin"] - 0.03, 0.08)

    total_debt = metrics.get("total_debt", 0)
    market_cap = metrics.get("market_cap", 1)
    if market_cap > 0:
        assumptions.debt_to_capital = total_debt / (total_debt + market_cap)
        assumptions.debt_to_capital = min(max(assumptions.debt_to_capital, 0.05), 0.70)

    sector = detect_sector(company.ticker)
    assumptions.ev_ebitda_multiple = MarketDataProvider.get_sector_multiple(
        company.ticker, "ev_ebitda", sector
    )
    assumptions.pe_multiple = MarketDataProvider.get_sector_multiple(
        company.ticker, "pe", sector
    )
    assumptions.ev_revenue_multiple = MarketDataProvider.get_sector_multiple(
        company.ticker, "ev_revenue", sector
    )
    assumptions.transaction_ev_ebitda = assumptions.ev_ebitda_multiple * 1.15

    return assumptions

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_valuation(company, sector, assumptions_dict):
    """Run all valuations with caching"""
    val_model = ValuationModel(company, sector)
    
    dcf_kwargs = {
        "stage1_years": assumptions_dict.get("stage1_years", 3),
        "stage2_years": assumptions_dict.get("stage2_years", 4),
        "stage1_growth": assumptions_dict.get("stage1_growth", 0.12),
        "terminal_growth": assumptions_dict.get("terminal_growth", 0.03),
        "wacc": assumptions_dict.get("wacc"),
        "risk_free_rate": assumptions_dict.get("risk_free_rate", 0.065),
        "equity_risk_premium": assumptions_dict.get("equity_risk_premium", 0.055),
        "stage1_ebitda_margin": assumptions_dict.get("stage1_ebitda_margin"),
        "terminal_margin": assumptions_dict.get("terminal_margin"),
        "reinvestment_rate": assumptions_dict.get("reinvestment_rate", 0.40),
        "tax_rate": assumptions_dict.get("tax_rate", 0.25),
        "use_mid_year_discount": True
    }
    
    comps_kwargs = {
        "custom_ev_ebitda": assumptions_dict.get("ev_ebitda_multiple"),
        "custom_pe": assumptions_dict.get("pe_multiple"),
        "custom_ev_revenue": assumptions_dict.get("ev_revenue_multiple"),
        "custom_pb": assumptions_dict.get("pb_multiple")
    }
    
    precedent_kwargs = {
        "custom_ev_ebitda": assumptions_dict.get("transaction_ev_ebitda"),
        "custom_ev_revenue": assumptions_dict.get("transaction_ev_revenue"),
        "custom_control_premium": assumptions_dict.get("control_premium")
    }
    
    try:
        dcf = val_model.run_dcf(**dcf_kwargs)
        comps = val_model.calculate_trading_comps(**comps_kwargs)
        precedent = val_model.calculate_precedent_transactions(**precedent_kwargs)
        
        return {
            "dcf": dcf,
            "comps": comps,
            "precedent": precedent,
            "val_model": val_model
        }
    except Exception as e:
        render_banner("error", "Valuation Failed", escape(str(e)))
        st.code(traceback.format_exc())
        return None

def compute_accretion_data(m1, m2, offer_premium, cash_pct, synergy_annual_total, tax_rate, ppa_settings=None):
    """Compute accretion/dilution metrics - moved outside tab4 for global availability"""
    ppa_settings = ppa_settings or {}

    # Get shares with proper fallbacks
    acquirer_shares = m1.get("shares_outstanding")
    if not acquirer_shares or acquirer_shares <= 0:
        market_cap = m1.get("market_cap")
        current_price = m1.get("current_price")
        if market_cap and market_cap > 0 and current_price and current_price > 0:
            acquirer_shares = market_cap / current_price
        else:
            acquirer_shares = 1e9
            render_banner("warning", "Estimated Shares", f"Using estimated shares for acquirer: {acquirer_shares:,.0f}")
    
    # Get acquirer EPS - prefer direct EPS from data_fetcher
    acquirer_eps = m1.get("eps")
    if not acquirer_eps or acquirer_eps <= 0:
        net_income = m1.get("net_income", 0)
        if net_income > 0 and acquirer_shares > 0:
            acquirer_eps = net_income / acquirer_shares
        else:
            acquirer_eps = 100  # Fallback
    
    # Get target metrics
    target_market_cap = m2.get("market_cap", 100e9)
    if not target_market_cap or target_market_cap <= 0:
        target_market_cap = 100e9
    
    # Offer value
    offer_value = target_market_cap * (1 + offer_premium / 100)
    
    # Synergy after tax
    synergy_after_tax = synergy_annual_total * (1 - tax_rate)

    # PPA and financing drag on earnings (after-tax)
    acquirer_cash = m1.get("cash", 0) or 0
    cash_portion = offer_value * (cash_pct / 100)
    available_acquirer_cash = acquirer_cash * 0.4
    cash_from_acquirer = min(cash_portion, available_acquirer_cash)
    new_debt_needed = max(0, cash_portion - cash_from_acquirer)

    intangibles_write_up_pct = float(ppa_settings.get("intangibles_write_up_pct", 15.0)) / 100
    useful_life_years = max(1, int(ppa_settings.get("useful_life_years", 10)))
    interest_rate_pct = float(ppa_settings.get("interest_rate_pct", 9.0))
    interest_rate = interest_rate_pct / 100

    annual_amortization = (offer_value * intangibles_write_up_pct) / useful_life_years
    annual_interest = new_debt_needed * interest_rate
    ppa_drag_after_tax = (annual_amortization + annual_interest) * (1 - tax_rate)
    
    # Pro-forma net income
    pro_forma_ni = m1.get("net_income", 0) + m2.get("net_income", 0) + synergy_after_tax - ppa_drag_after_tax
    
    # Stock consideration
    acquirer_price = m1.get("current_price")
    if not acquirer_price or acquirer_price <= 0:
        acquirer_price = 1000
    
    stock_portion = offer_value * (1 - cash_pct / 100)
    new_shares = stock_portion / acquirer_price if acquirer_price > 0 else 0
    total_shares = acquirer_shares + new_shares
    
    # Pro-forma EPS
    pro_forma_eps = pro_forma_ni / total_shares if total_shares > 0 else 0
    
    # Accretion/Dilution
    accretion = ((pro_forma_eps / acquirer_eps) - 1) * 100 if acquirer_eps > 0 else 0
    
    return {
        "acquirer_shares": acquirer_shares,
        "acquirer_eps": acquirer_eps,
        "target_market_cap": target_market_cap,
        "offer_value": offer_value,
        "synergy_after_tax": synergy_after_tax,
        "ppa_drag_after_tax": ppa_drag_after_tax,
        "annual_amortization": annual_amortization,
        "annual_interest": annual_interest,
        "pro_forma_ni": pro_forma_ni,
        "acquirer_price": acquirer_price,
        "stock_portion": stock_portion,
        "new_shares": new_shares,
        "total_shares": total_shares,
        "pro_forma_eps": pro_forma_eps,
        "accretion": accretion
    }

def build_ppa_context(company1, company2, offer_premium, cash_pct, tax_rate):
    """Build PPA and pro forma inputs shared by the UI and PDF report."""
    acquirer_bs = company1.get_balance_sheet_metrics() if hasattr(company1, "get_balance_sheet_metrics") else {}
    target_bs = company2.get_balance_sheet_metrics() if hasattr(company2, "get_balance_sheet_metrics") else {}
    acquirer_metrics = company1.get_key_metrics()
    target_metrics = company2.get_key_metrics()

    target_market_cap = target_metrics.get("market_cap", 100e9)
    offer_value = target_market_cap * (1 + offer_premium / 100)

    target_book_value = target_bs.get("book_value")
    if not target_book_value or target_book_value <= 0:
        pb_ratio = target_metrics.get("pb_ratio")
        if pb_ratio and pb_ratio > 0:
            target_book_value = target_market_cap / pb_ratio
        else:
            target_book_value = target_market_cap / 2

    target_tangible_book = target_bs.get("tangible_book_value", target_book_value * 0.7)

    ppa_result = calculate_goodwill_and_ppa(
        purchase_price=offer_value,
        target_book_value=target_book_value,
        target_tangible_book_value=target_tangible_book,
        ppe_write_up_pct=float(st.session_state.ppa_ppe_write_up_pct) / 100,
        intangibles_write_up_pct=float(st.session_state.ppa_intangibles_write_up_pct) / 100,
    )

    cash_pct_decimal = cash_pct / 100
    stock_pct_decimal = 1 - cash_pct_decimal
    acquirer_cash = acquirer_bs.get("cash", acquirer_metrics.get("cash", 0)) or 0
    target_debt = target_bs.get("total_debt", target_metrics.get("total_debt", 0)) or 0
    cash_portion = offer_value * cash_pct_decimal
    stock_portion = offer_value * stock_pct_decimal
    available_acquirer_cash = acquirer_cash * 0.4
    cash_from_acquirer = min(cash_portion, available_acquirer_cash)
    new_debt_needed = max(0, cash_portion - cash_from_acquirer)

    financing = {
        "cash_from_acquirer": cash_from_acquirer,
        "new_debt": new_debt_needed,
        "stock_consideration": stock_portion,
    }

    sources_df, uses_df = create_sources_uses_table(
        offer_value,
        cash_pct,
        stock_pct_decimal * 100,
        acquirer_cash,
        target_debt,
        float(st.session_state.ppa_transaction_fees_pct) / 100,
        0.005,
    )

    pro_forma_df = create_pro_forma_balance_sheet(
        acquirer_bs,
        target_bs,
        ppa_result,
        financing,
        tax_rate,
    )

    useful_life_years = int(st.session_state.ppa_useful_life_years)
    annual_amortization = ppa_result["intangibles_write_up"] / useful_life_years if useful_life_years > 0 else 0
    annual_amortization_after_tax = annual_amortization * (1 - tax_rate)
    interest_rate = float(st.session_state.ppa_interest_rate_pct) / 100
    annual_interest = financing["new_debt"] * interest_rate
    annual_interest_after_tax = annual_interest * (1 - tax_rate)

    return {
        "acquirer_bs": acquirer_bs,
        "target_bs": target_bs,
        "acquirer_metrics": acquirer_metrics,
        "target_metrics": target_metrics,
        "target_market_cap": target_market_cap,
        "offer_value": offer_value,
        "target_book_value": target_book_value,
        "target_tangible_book": target_tangible_book,
        "ppa_result": ppa_result,
        "financing": financing,
        "sources_df": sources_df,
        "uses_df": uses_df,
        "pro_forma_df": pro_forma_df,
        "useful_life_years": useful_life_years,
        "annual_amortization": annual_amortization,
        "annual_amortization_after_tax": annual_amortization_after_tax,
        "interest_rate": interest_rate,
        "annual_interest": annual_interest,
        "annual_interest_after_tax": annual_interest_after_tax,
        "total_annual_impact": -annual_amortization_after_tax - annual_interest_after_tax,
    }


def build_memo_context(dcf2, synergies, accretion_value, offer_premium):
    """Build recommendation content for the report."""
    if accretion_value > 5 and dcf2.implied_premium > -20:
        recommendation = "PROCEED"
        rec_color = "success"
    elif accretion_value > 0 or dcf2.implied_premium > -30:
        recommendation = "CONSIDER WITH CAUTION"
        rec_color = "warning"
    else:
        recommendation = "DO NOT PROCEED"
        rec_color = "error"

    if dcf2.implied_premium < -15 and accretion_value < 10:
        recommendation = "CONSIDER WITH CAUTION"
        rec_color = "warning"

    fairness_text = dcf2.fairness_rating.lower() if hasattr(dcf2, "fairness_rating") else "fairly valued"
    key_drivers = [
        f"Valuation: Target is {fairness_text} vs current price ({dcf2.implied_premium:+.0f}% premium)",
        f"Accretion: Deal is {accretion_value:+.1f}% to Acquirer EPS",
        f"Synergies: Identified Rs.{synergies.get('pv_total', 0)/1e9:.1f}B in PV synergies",
        f"Multiple: Offer represents {offer_premium:.0f}% premium to current price",
    ]

    risks = pd.DataFrame({
        "Risk": ["Integration Execution", "Synergy Realization", "Regulatory", "Market Reaction"],
        "Mitigation": [
            "Phased integration with dedicated team",
            "Conservative targets with contingency",
            "Early regulatory engagement",
            "Clear strategic communication",
        ],
    })

    return {
        "recommendation": recommendation,
        "rec_color": rec_color,
        "fairness_text": fairness_text,
        "key_drivers": key_drivers,
        "risks": risks,
    }


def create_sources_uses_table(offer_value: float, cash_pct: float, stock_pct: float,
                              acquirer_cash: float, target_debt: float,
                              transaction_fees_pct: float = 0.01,
                              deal_costs_pct: float = 0.005) -> pd.DataFrame:
    """Create Sources & Uses table for M&A transaction"""

    # Calculate components
    cash_portion = offer_value * (cash_pct / 100)
    stock_portion = offer_value * (stock_pct / 100)

    # Assume acquirer uses 40% of available cash, rest is new debt
    available_acquirer_cash = acquirer_cash * 0.4  # Don't deplete all cash
    new_debt_needed = max(0, cash_portion - available_acquirer_cash)
    cash_from_acquirer = min(cash_portion, available_acquirer_cash)

    # Sources
    sources = {
        "Source": [
            "Cash from Acquirer Balance Sheet",
            "Stock Consideration (New Shares)",
            "New Debt Financing",
            "Total Sources"
        ],
        "Amount (Rs.B)": [
            cash_from_acquirer / 1e9,
            stock_portion / 1e9,
            new_debt_needed / 1e9,
            offer_value / 1e9
        ],
        "% of Total": [
            (cash_from_acquirer / offer_value * 100) if offer_value else 0,
            (stock_portion / offer_value * 100) if offer_value else 0,
            (new_debt_needed / offer_value * 100) if offer_value else 0,
            100
        ]
    }

    # Uses
    transaction_fees = offer_value * transaction_fees_pct
    deal_costs = offer_value * deal_costs_pct

    uses = {
        "Use": [
            "Equity Purchase Price",
            "Transaction Fees (Legal, Advisory)",
            "Deal Costs (Filing, Regulatory)",
            "Debt Repayment (Target)",
            "Total Uses"
        ],
        "Amount (Rs.B)": [
            offer_value / 1e9,
            transaction_fees / 1e9,
            deal_costs / 1e9,
            target_debt / 1e9,
            (offer_value + transaction_fees + deal_costs + target_debt) / 1e9
        ]
    }

    return pd.DataFrame(sources), pd.DataFrame(uses)


def calculate_goodwill_and_ppa(purchase_price: float, target_book_value: float,
                               target_tangible_book_value: float,
                               ppe_write_up_pct: float = 0.10,
                               intangibles_write_up_pct: float = 0.15) -> dict:
    """Calculate Goodwill and Purchase Price Allocation"""

    # Ensure we have valid numbers
    target_tangible_book_value = target_tangible_book_value or target_book_value

    # Calculate write-ups
    ppe_write_up = purchase_price * ppe_write_up_pct
    intangibles_write_up = purchase_price * intangibles_write_up_pct

    # Adjusted tangible book value after write-ups
    adjusted_tangible_book = target_tangible_book_value + ppe_write_up

    # Goodwill = Purchase Price - (Tangible Book Value + Identifiable Intangibles)
    goodwill = max(0, purchase_price - (adjusted_tangible_book + intangibles_write_up))

    return {
        "purchase_price": purchase_price,
        "target_book_value": target_book_value,
        "target_tangible_book_value": target_tangible_book_value,
        "ppe_write_up": ppe_write_up,
        "intangibles_write_up": intangibles_write_up,
        "adjusted_tangible_book": adjusted_tangible_book,
        "goodwill": goodwill,
        "total_allocated": adjusted_tangible_book + intangibles_write_up + goodwill
    }


def create_pro_forma_balance_sheet(acquirer_bs: dict, target_bs: dict,
                                   ppa_result: dict, financing: dict,
                                   tax_rate: float = 0.25) -> pd.DataFrame:
    """Create pro forma combined balance sheet"""

    def _num(value):
        try:
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    # Extract values with defaults
    acquirer_assets = _num(acquirer_bs.get("total_assets"))
    acquirer_liabilities = _num(acquirer_bs.get("total_liabilities"))
    acquirer_equity = acquirer_assets - acquirer_liabilities

    target_assets = _num(target_bs.get("total_assets"))
    target_liabilities = _num(target_bs.get("total_liabilities"))
    target_equity = target_assets - target_liabilities

    # Purchase price and financing
    purchase_price = _num(ppa_result["purchase_price"])
    cash_from_acquirer = _num(financing.get("cash_from_acquirer"))
    new_debt = _num(financing.get("new_debt"))
    stock_consideration = _num(financing.get("stock_consideration"))

    # Pro forma adjustments
    # 1. Cash decreases by cash used for acquisition
    pro_forma_cash = (_num(acquirer_bs.get("cash")) + _num(target_bs.get("cash"))) - cash_from_acquirer

    # 2. Goodwill increases (new asset)
    new_goodwill = _num(ppa_result["goodwill"])

    # 3. Intangibles increase (write-up)
    new_intangibles = _num(ppa_result["intangibles_write_up"])

    # 4. PP&E increases (write-up)
    ppe_increase = _num(ppa_result["ppe_write_up"])

    # 5. New debt added
    pro_forma_debt = _num(acquirer_bs.get("total_debt")) + _num(target_bs.get("total_debt")) + new_debt

    # 6. Equity adjustment (new shares issued)
    pro_forma_equity = acquirer_equity + target_equity + stock_consideration

    # Calculate deferred tax liability from write-ups (taxable temporary differences)
    deferred_tax_liability = (ppe_increase + new_intangibles) * tax_rate

    acquirer_cash = _num(acquirer_bs.get("cash"))
    target_cash = _num(target_bs.get("cash"))
    acquirer_receivables = _num(acquirer_bs.get("receivables"))
    target_receivables = _num(target_bs.get("receivables"))
    acquirer_inventory = _num(acquirer_bs.get("inventory"))
    target_inventory = _num(target_bs.get("inventory"))
    acquirer_ppe = _num(acquirer_bs.get("ppe"))
    target_ppe = _num(target_bs.get("ppe"))
    acquirer_intangibles = _num(acquirer_bs.get("intangible_assets"))
    target_intangibles = _num(target_bs.get("intangible_assets"))
    acquirer_goodwill = _num(acquirer_bs.get("goodwill"))
    target_goodwill = _num(target_bs.get("goodwill"))
    acquirer_payables = _num(acquirer_bs.get("payables"))
    target_payables = _num(target_bs.get("payables"))
    acquirer_other_liabilities = _num(acquirer_bs.get("other_liabilities"))
    target_other_liabilities = _num(target_bs.get("other_liabilities"))

    # Build pro forma balance sheet
    pro_forma_data = {
        "Category": ["Assets", "Assets", "Assets", "Assets", "Assets", "Assets", "", "Liabilities & Equity", "Liabilities & Equity", "Liabilities & Equity", "Liabilities & Equity", "Liabilities & Equity", "Liabilities & Equity"],
        "Item": [
            "Cash & Equivalents",
            "Accounts Receivable",
            "Inventory",
            "PP&E (Adjusted)",
            "Identifiable Intangibles",
            "Goodwill",
            "Total Assets",
            "Total Debt",
            "Accounts Payable",
            "Deferred Tax Liability",
            "Other Liabilities",
            "Total Equity",
            "Total Liabilities & Equity"
        ],
        "Acquirer (Rs.B)": [
            acquirer_cash / 1e9,
            acquirer_receivables / 1e9,
            acquirer_inventory / 1e9,
            acquirer_ppe / 1e9,
            acquirer_intangibles / 1e9,
            acquirer_goodwill / 1e9,
            (acquirer_assets / 1e9),
            _num(acquirer_bs.get("total_debt")) / 1e9,
            acquirer_payables / 1e9,
            0,
            acquirer_other_liabilities / 1e9,
            (acquirer_equity / 1e9),
            (acquirer_assets / 1e9)
        ],
        "Target (Rs.B)": [
            target_cash / 1e9,
            target_receivables / 1e9,
            target_inventory / 1e9,
            target_ppe / 1e9,
            target_intangibles / 1e9,
            target_goodwill / 1e9,
            (target_assets / 1e9),
            _num(target_bs.get("total_debt")) / 1e9,
            target_payables / 1e9,
            0,
            target_other_liabilities / 1e9,
            (target_equity / 1e9),
            (target_assets / 1e9)
        ],
        "Pro Forma Adjustments (Rs.B)": [
            -cash_from_acquirer / 1e9,
            0,
            0,
            ppe_increase / 1e9,
            new_intangibles / 1e9,
            new_goodwill / 1e9,
            (ppe_increase + new_intangibles + new_goodwill - cash_from_acquirer) / 1e9,
            new_debt / 1e9,
            0,
            deferred_tax_liability / 1e9,
            0,
            stock_consideration / 1e9,
            (new_debt + deferred_tax_liability + stock_consideration) / 1e9
        ],
        "Pro Forma Combined (Rs.B)": [
            (acquirer_cash + target_cash - cash_from_acquirer) / 1e9,
            (acquirer_receivables + target_receivables) / 1e9,
            (acquirer_inventory + target_inventory) / 1e9,
            (acquirer_ppe + target_ppe + ppe_increase) / 1e9,
            (acquirer_intangibles + target_intangibles + new_intangibles) / 1e9,
            (acquirer_goodwill + target_goodwill + new_goodwill) / 1e9,
            (acquirer_assets + target_assets + ppe_increase + new_intangibles + new_goodwill - cash_from_acquirer) / 1e9,
            (_num(acquirer_bs.get("total_debt")) + _num(target_bs.get("total_debt")) + new_debt) / 1e9,
            (acquirer_payables + target_payables) / 1e9,
            deferred_tax_liability / 1e9,
            (acquirer_other_liabilities + target_other_liabilities) / 1e9,
            (acquirer_equity + target_equity + stock_consideration) / 1e9,
            (acquirer_assets + target_assets + ppe_increase + new_intangibles + new_goodwill - cash_from_acquirer) / 1e9
        ]
    }

    return pd.DataFrame(pro_forma_data)


def create_ppa_summary_table(ppa_result: dict) -> pd.DataFrame:
    """Create Purchase Price Allocation summary table"""
    data = {
        "Component": [
            "Purchase Price",
            "Less: Target Tangible Book Value",
            "Plus: PP&E Write-up",
            "Equals: Adjusted Tangible Book Value",
            "Less: Identifiable Intangibles (Brand, Customer Relationships, etc.)",
            "Equals: Goodwill"
        ],
        "Amount (Rs.B)": [
            ppa_result["purchase_price"] / 1e9,
            -ppa_result["target_tangible_book_value"] / 1e9,
            ppa_result["ppe_write_up"] / 1e9,
            ppa_result["adjusted_tangible_book"] / 1e9,
            -ppa_result["intangibles_write_up"] / 1e9,
            ppa_result["goodwill"] / 1e9
        ]
    }
    return pd.DataFrame(data)

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<p class="header">M&A Intelligence Platform</p>', unsafe_allow_html=True)
with col2:
    st.caption("Institutional-Grade Analysis")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class='app-shell-title'>
            <div class='app-monogram'>MA</div>
            <div>
                <div class='app-title'>M&A Intelligence</div>
                <div class='app-subtitle'>Deal Configuration</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("Deal Configuration", "Acquirer / Target / Terms")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Acquirer</div>", unsafe_allow_html=True)
    ticker1 = st.text_input("Acquirer Ticker", "RELIANCE.NS", key="t1").upper().strip()
    sector1 = detect_sector(ticker1)
    sector1_badge = {
        "technology": "badge-blue",
        "financial": "badge-green",
        "energy": "badge-amber",
    }.get(sector1, "")
    st.markdown(
        f"<div class='badge-pill {sector1_badge}'>Sector: {escape(sector1.title())}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Target</div>", unsafe_allow_html=True)
    ticker2 = st.text_input("Target Ticker", "INFY.NS", key="t2").upper().strip()
    sector2 = detect_sector(ticker2)
    sector2_badge = {
        "technology": "badge-blue",
        "financial": "badge-green",
        "energy": "badge-amber",
    }.get(sector2, "")
    st.markdown(
        f"<div class='badge-pill {sector2_badge}'>Sector: {escape(sector2.title())}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Deal Terms</div>", unsafe_allow_html=True)

    control_premium_pct = float(st.session_state.assumptions.control_premium * 100)
    if "offer_premium_pct" not in st.session_state:
        st.session_state.offer_premium_pct = control_premium_pct
    if "offer_premium_source_assumption" not in st.session_state:
        st.session_state.offer_premium_source_assumption = control_premium_pct
    if abs(st.session_state.offer_premium_source_assumption - control_premium_pct) > 1e-6:
        st.session_state.offer_premium_pct = control_premium_pct
        st.session_state.offer_premium_source_assumption = control_premium_pct

    offer_premium = st.slider("Offer Premium (%)", 0, 80, int(round(st.session_state.offer_premium_pct)), key="offer_premium_pct", on_change=invalidate_cached_valuation)
    cash_pct = st.slider("Cash Consideration (%)", 0, 100, 60, on_change=invalidate_cached_valuation)
    st.session_state.assumptions.control_premium = offer_premium / 100
    st.session_state.offer_premium_source_assumption = offer_premium
    
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    loaded_company1 = st.session_state.get("loaded_company1")
    loaded_company2 = st.session_state.get("loaded_company2")
    if loaded_company1 and loaded_company2 and st.session_state.get("last_tickers") == (ticker1, ticker2):
        render_banner("info", "AI Ready", "AI assumptions are available in the Assumptions tab for the current deal pair.")
    else:
        render_banner("info", "Ready", "Run analysis to enable AI assumptions and valuation context.")
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with custom_loading("Loading data and running valuations..."):
            st.session_state.last_tickers = (ticker1, ticker2)
            st.session_state.run_analysis = True
            st.session_state.valuation_results = None  # Clear cache
            st.session_state.accretion_data = None  # Clear accretion cache

    if st.session_state.get("accretion_data") and st.session_state.get("run_analysis"):
        acc_sidebar = st.session_state.accretion_data
        render_banner(
            "info",
            "Deal Summary",
            f"Offer value <strong>{format_money(acc_sidebar['offer_value'] / 1e9, 1)}</strong>, premium <strong>{offer_premium:.0f}%</strong>, accretion <strong>{acc_sidebar['accretion']:+.1f}%</strong>.",
        )

# Main Analysis
if st.session_state.get("run_analysis", False):
    ticker1, ticker2 = st.session_state.last_tickers
    
    if not ticker1 or not ticker2:
        render_banner("error", "Missing Tickers", "Please enter both tickers.")
        st.stop()
    
    # Load companies
    with custom_loading(f"Loading {ticker1} and {ticker2}..."):
        try:
            company1 = get_company(ticker1)
            company2 = get_company(ticker2)
            
            if not company1.success:
                render_banner("error", f"Failed to load {ticker1}", escape(str(company1.error)))
                st.stop()
            if not company2.success:
                render_banner("error", f"Failed to load {ticker2}", escape(str(company2.error)))
                st.stop()
                
        except Exception as e:
            render_banner("error", "Data Load Failed", escape(str(e)))
            st.stop()
    
    render_banner(
        "success",
        "Loaded",
        f"{escape(company1.get_key_metrics()['name'])} &amp; {escape(company2.get_key_metrics()['name'])} are ready for analysis.",
    )
    st.session_state.loaded_company1 = company1
    st.session_state.loaded_company2 = company2
    st.session_state.loaded_sector1 = sector1
    st.session_state.loaded_sector2 = sector2
    
    # Derive assumptions for target
    if company2.success and st.session_state.assumptions_source_ticker != ticker2:
        st.session_state.assumptions = derive_assumptions_from_company(
            company2, st.session_state.assumptions
        )
        st.session_state.assumptions_source_ticker = ticker2
    
    # Seed synergy values from the target company so PDF generation has
    # meaningful defaults even before the Synergies tab is opened.
    synergy_values = st.session_state.get("synergy_values")
    if not synergy_values or all(float(value or 0) == 0 for value in synergy_values.values()):
        target_metrics = company2.get_key_metrics() if company2.success else {}
        target_rev = target_metrics.get("revenue", 0) or 0
        if target_rev <= 0:
            target_rev = 100e9
        target_costs = target_rev * 0.75
        default_rev_syn = target_rev * 0.10
        default_cost_syn = target_costs * 0.15
        st.session_state.synergy_values = {
            "annual_rev": default_rev_syn,
            "annual_cost": default_cost_syn,
            "annual_total": default_rev_syn + default_cost_syn,
            "pv_rev": default_rev_syn * 6,
            "pv_cost": default_cost_syn * 6,
            "pv_total": (default_rev_syn + default_cost_syn) * 6,
        }
    
    # Run valuations if not cached
    if st.session_state.valuation_results is None:
        assumptions_dict = {
            "stage1_years": st.session_state.assumptions.stage1_years,
            "stage2_years": st.session_state.assumptions.stage2_years,
            "stage1_growth": st.session_state.assumptions.stage1_growth,
            "terminal_growth": st.session_state.assumptions.terminal_growth,
            "risk_free_rate": st.session_state.assumptions.risk_free_rate,
            "equity_risk_premium": st.session_state.assumptions.equity_risk_premium,
            "tax_rate": st.session_state.assumptions.tax_rate,
            "stage1_ebitda_margin": st.session_state.assumptions.stage1_ebitda_margin,
            "terminal_margin": st.session_state.assumptions.terminal_margin,
            "reinvestment_rate": st.session_state.assumptions.reinvestment_rate,
            "debt_to_capital": st.session_state.assumptions.debt_to_capital,
            "cost_of_debt_spread": st.session_state.assumptions.cost_of_debt_spread,
            "ev_ebitda_multiple": st.session_state.assumptions.ev_ebitda_multiple,
            "pe_multiple": st.session_state.assumptions.pe_multiple,
            "ev_revenue_multiple": st.session_state.assumptions.ev_revenue_multiple,
            "pb_multiple": st.session_state.assumptions.pb_multiple,
            "transaction_ev_revenue": st.session_state.assumptions.transaction_ev_revenue,
            "transaction_ev_ebitda": st.session_state.assumptions.transaction_ev_ebitda,
            "control_premium": st.session_state.assumptions.control_premium,
        }

        debt_to_capital = assumptions_dict.get("debt_to_capital", 0.30)
        debt_to_capital = min(max(debt_to_capital, 0.0), 0.95)
        debt_to_equity = debt_to_capital / (1 - debt_to_capital) if debt_to_capital < 1 else 0.0
        cost_of_debt = assumptions_dict["risk_free_rate"] + assumptions_dict.get("cost_of_debt_spread", 0.02)

        acquirer_model = ValuationModel(company1, sector1)
        acquirer_wacc = acquirer_model.dcf_model.calculate_wacc(
            risk_free_rate=st.session_state.assumptions.risk_free_rate,
            equity_risk_premium=st.session_state.assumptions.equity_risk_premium,
            debt_to_equity=debt_to_equity,
            cost_of_debt=cost_of_debt,
            tax_rate=st.session_state.assumptions.tax_rate,
        )
        target_model = ValuationModel(company2, sector2)
        target_wacc = target_model.dcf_model.calculate_wacc(
            risk_free_rate=st.session_state.assumptions.risk_free_rate,
            equity_risk_premium=st.session_state.assumptions.equity_risk_premium,
            debt_to_equity=debt_to_equity,
            cost_of_debt=cost_of_debt,
            tax_rate=st.session_state.assumptions.tax_rate,
        )

        acquirer_assumptions_dict = dict(assumptions_dict)
        acquirer_assumptions_dict["wacc"] = acquirer_wacc
        target_assumptions_dict = dict(assumptions_dict)
        target_assumptions_dict["wacc"] = target_wacc
        
        with custom_loading("Running DCF and comparable analysis..."):
            val1 = run_valuation(company1, sector1, acquirer_assumptions_dict)
            val2 = run_valuation(company2, sector2, target_assumptions_dict)
            
            st.session_state.valuation_results = {
                "acquirer": val1,
                "target": val2
            }
    
    results = st.session_state.valuation_results
    
    if not results["acquirer"] or not results["target"]:
        render_banner("error", "Valuation Failed", "Check console for details.")
        st.stop()

    dcf1 = results["acquirer"]["dcf"]
    comps1 = results["acquirer"]["comps"]
    precedent1 = results["acquirer"]["precedent"]
    dcf2 = results["target"]["dcf"]
    comps2 = results["target"]["comps"]
    precedent2 = results["target"]["precedent"]
    val2 = results["target"]["val_model"]
    
    # Get metrics for display
    m1 = company1.get_key_metrics()
    m2 = company2.get_key_metrics()
    current_price2 = m2.get("current_price", 0)
    
    # ============================================================
    # CRITICAL FIX: Compute accretion data BEFORE tabs
    # This ensures PDF download always has correct data
    # ============================================================
    synergy_annual_total = st.session_state.synergy_values.get("annual_total", 0)
    tax_rate = st.session_state.assumptions.tax_rate
    ppa_settings = {
        "intangibles_write_up_pct": st.session_state.ppa_intangibles_write_up_pct,
        "useful_life_years": st.session_state.ppa_useful_life_years,
        "interest_rate_pct": st.session_state.ppa_interest_rate_pct,
    }
    
    accretion_data = compute_accretion_data(
        m1, m2, offer_premium, cash_pct, synergy_annual_total, tax_rate, ppa_settings
    )
    
    # Store in session state for PDF generation
    st.session_state.accretion_data = accretion_data
    
    # Extract values for easy access
    acquirer_eps = accretion_data["acquirer_eps"]
    acquirer_shares = accretion_data["acquirer_shares"]
    pro_forma_eps = accretion_data["pro_forma_eps"]
    accretion = accretion_data["accretion"]
    offer_value = accretion_data["offer_value"]

    render_status_bar(
        m1.get("name", ticker1),
        m2.get("name", ticker2),
        offer_value / 1e9,
        offer_premium,
    )
    
    # ============================================================
    # PDF Download Button (now uses pre-computed accretion_data and active assumptions)
    # ============================================================
    col_download1, col_download2 = st.columns([1, 5])
    with col_download1:
        if "pdf_report_bytes" not in st.session_state:
            st.session_state.pdf_report_bytes = None
            st.session_state.pdf_report_filename = None

        def clear_pdf_download_state():
            st.session_state.pdf_report_bytes = None
            st.session_state.pdf_report_filename = None

        if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
            with custom_loading("Generating PDF report..."):
                try:
                    synergy_values = st.session_state.synergy_values
                    deal_terms = {
                        "premium": offer_premium,
                        "cash_pct": cash_pct
                    }

                    # Get the pre-computed accretion from session state
                    acc_data = st.session_state.accretion_data
                    accretion_value = acc_data["accretion"] if acc_data else 0
                    pro_forma_eps_value = acc_data["pro_forma_eps"] if acc_data else 0
                    acquirer_eps_value = acc_data["acquirer_eps"] if acc_data else 0

                    # Build PPA & memo contexts
                    ppa_context = build_ppa_context(company1, company2, offer_premium, cash_pct, tax_rate)
                    memo_context = build_memo_context(dcf2, st.session_state.synergy_values, accretion_value, offer_premium)

                    current_assumptions = st.session_state.assumptions

                    report_context = {
                        "ppa": ppa_context,
                        "memo": memo_context,
                        "synergies": synergy_values,
                        "accretion_data": acc_data,
                        "ai_active": st.session_state.get("ai_assumptions_active", False),
                        "ai_rationales": st.session_state.get("ai_rationales", {}),
                        "ai_confidence": st.session_state.get("ai_confidence", None),
                        "ai_key_risks": st.session_state.get("ai_key_risks", []),
                        "ai_mode": st.session_state.get("ai_mode", "base"),
                    }

                    report_gen = MAReportGenerator(
                        company1=company1,
                        company2=company2,
                        dcf1=dcf1,
                        dcf2=dcf2,
                        comps1=comps1,
                        comps2=comps2,
                        precedent1=precedent1,
                        precedent2=precedent2,
                        val_model=val2,
                        assumptions=current_assumptions,
                        deal_terms=deal_terms,
                        synergies=synergy_values,
                        accretion=accretion_value,
                        report_context=report_context,
                    )

                    report_gen.acquirer_eps = acquirer_eps_value
                    report_gen.pro_forma_eps = pro_forma_eps_value

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        report_path = tmp_file.name

                    report_gen.generate(report_path)

                    with open(report_path, "rb") as f:
                        st.session_state.pdf_report_bytes = f.read()
                    st.session_state.pdf_report_filename = f"M&A_Report_{company2.ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                    os.unlink(report_path)
                except Exception as e:
                    st.session_state.pdf_report_bytes = None
                    st.session_state.pdf_report_filename = None
                    render_banner("error", "Report Generation Failed", escape(str(e)))
                    render_banner("info", "Dependency Note", "Make sure you have installed: pip install reportlab pillow plotly kaleido")

        if st.session_state.pdf_report_bytes and st.session_state.pdf_report_filename:
            st.download_button(
                label="💾 Download PDF Report",
                data=st.session_state.pdf_report_bytes,
                file_name=st.session_state.pdf_report_filename,
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download_button",
                on_click=clear_pdf_download_state,
            )
    
    # Main view selector.
    # Streamlit tabs reset on every rerun, so use a persisted selector instead.
    selected_view = st.radio(
        "Navigation",
        ["📊 Overview", "📈 Valuation", "🤝 Synergies", "💰 Deal Mechanics", "🏗️ PPA & Pro Forma", "📝 Memo", "📌 Assumptions"],
        horizontal=True,
        key="main_view",
        label_visibility="collapsed",
    )

    if selected_view == "📊 Overview":
        ai_active = st.session_state.assumptions.has_ai_assumptions()

        render_section_header(f"{m2['name']} - Financial Overview", "Overview")

        if ai_active:
            render_banner("success", "AI Active", "Values below reflect AI-generated inputs.")

            with st.expander("📋 View AI-Driven Assumptions", expanded=False):
                ai_rationales = st.session_state.assumptions.get_ai_rationales()
                st.markdown("**Key AI Rationales:**")
                for key, rationale in list(ai_rationales.items())[:5]:
                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {rationale[:200]}...")

        # Display key metrics
        render_metric_grid([
            {"label": f"{m1['name']} - Market Cap", "value": f"₹{m1.get('market_cap', 0)/1e9:.1f}B"},
            {"label": "Revenue (TTM)", "value": f"₹{m1.get('revenue', 0)/1e9:.1f}B"},
            {"label": "EBITDA Margin", "value": f"{m1.get('ebitda_margin', 0)*100:.1f}%"},
            {"label": f"{m2['name']} - Market Cap", "value": f"₹{m2.get('market_cap', 0)/1e9:.1f}B"},
            {"label": "Revenue (TTM)", "value": f"₹{m2.get('revenue', 0)/1e9:.1f}B"},
            {"label": "EBITDA Margin", "value": f"{m2.get('ebitda_margin', 0)*100:.1f}%"},
            {"label": "Current Price (Target)", "value": f"₹{current_price2:,.2f}"},
            {"label": "52W High/Low", "value": f"₹{m2.get('52w_high', 0):,.0f} / ₹{m2.get('52w_low', 0):,.0f}"},
            {"label": "P/E Ratio", "value": f"{m2.get('pe_ratio', 0):.1f}x"},
        ], columns=3)
        if m1.get('eps') or m2.get('eps'):
            render_metric_grid([
                {"label": "EPS (TTM) - Acquirer", "value": f"₹{m1.get('eps', 0):.2f}"},
                {"label": "EPS (TTM) - Target", "value": f"₹{m2.get('eps', 0):.2f}"},
            ], columns=2)

        if ai_active:
            st.caption(
                "📊 Key AI-driven assumptions used: "
                f"Stage1 Growth: {st.session_state.assumptions.stage1_growth:.1%}, "
                f"Terminal Growth: {st.session_state.assumptions.terminal_growth:.1%}, "
                f"EV/EBITDA: {st.session_state.assumptions.ev_ebitda_multiple:.1f}x"
            )
    
    if selected_view == "📈 Valuation":
        render_section_header(f"{m2['name']} - Valuation Analysis", "DCF Analysis")

        if st.session_state.assumptions.has_ai_assumptions():
            render_banner("info", "AI Active", "DCF and comps use AI-generated inputs.")
        
        val_model = results["target"]["val_model"]
        
        # Valuation summary cards
        avg = np.mean([dcf2.per_share, comps2.per_share_weighted, precedent2.per_share_with_premium])
        render_metric_grid([
            {
                "label": "DCF Value",
                "value": f"₹{dcf2.per_share:,.0f}",
                "delta": f"{dcf2.implied_premium:+.0f}%" if dcf2.implied_premium != 0 else None,
                "delta_state": "positive" if dcf2.implied_premium >= 0 else "negative",
            },
            {"label": "Trading Comps", "value": f"₹{comps2.per_share_weighted:,.0f}"},
            {"label": "Precedent Transactions", "value": f"₹{precedent2.per_share_with_premium:,.0f}"},
            {"label": "Weighted Average", "value": f"₹{avg:,.0f}"},
        ], columns=4)

        if st.session_state.assumptions.has_ai_assumptions():
            with st.expander("🤖 AI Valuation Rationale", expanded=False):
                ai_rationales = st.session_state.assumptions.get_ai_rationales()
                dcf_rationales = {
                    k: v
                    for k, v in ai_rationales.items()
                    if k in ["stage1_growth", "terminal_growth", "ev_ebitda_multiple", "control_premium"]
                }
                for key, rationale in dcf_rationales.items():
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {rationale}")
        
        # Terminal value warning if applicable
        terminal_pct = (dcf2.terminal_pv / dcf2.enterprise_value) * 100 if dcf2.enterprise_value > 0 else 0
        if terminal_pct > 70:
            render_banner("warning", "Terminal Value Concentration", f"Terminal value represents {terminal_pct:.0f}% of total DCF value. Terminal growth assumption is critical.")
        
        # Football field
        st.markdown("#### Valuation Football Field")
        try:
            fig = val_model.create_football_field(dcf2, comps2, precedent2)
            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displaylogo": False, "responsive": True, "displayModeBar": True},
                    key="football_field_chart"
                )
            else:
                st.info("Football field chart data is temporarily unavailable.")
        except Exception as e:
            st.warning(f"⚠️ Football Field chart failed to render in this environment: {e}")
            st.caption("This usually indicates a missing Plotly dependency on the deployment server.")
        
        # Sensitivity analysis
        with st.expander("Sensitivity Analysis", expanded=False):
            try:
                fig_sens = val_model.create_sensitivity_heatmap(dcf2)
                if fig_sens is not None:
                    st.plotly_chart(
                        fig_sens,
                        use_container_width=True,
                        config={"displaylogo": False, "responsive": True, "displayModeBar": True},
                        key="sensitivity_chart"
                    )
                else:
                    st.info("Sensitivity analysis data is temporarily unavailable.")
            except Exception as e:
                st.warning(f"⚠️ Sensitivity chart failed to render: {e}")
        
        # Detailed FCF table
        with st.expander("Detailed FCF Projections", expanded=False):
            fcf_table = pd.DataFrame({
                "Year": dcf2.projection_years,
                "Revenue (₹B)": [r/1e9 for r in dcf2.revenue_projections],
                "FCF (₹B)": [f/1e9 for f in dcf2.fcf_projections],
                "PV Factor": [1/(1+dcf2.wacc)**(i+0.5) for i in range(len(dcf2.projection_years))],
                "PV of FCF (₹B)": [(f/(1+dcf2.wacc)**(i+0.5))/1e9 for i, f in enumerate(dcf2.fcf_projections)]
            })
            render_table(fcf_table.round(2))
            
            render_banner("info", "Assumptions Used", f"WACC = {dcf2.wacc:.1%}, Terminal Growth = {dcf2.terminal_growth:.1%}, Mid-year discounting applied.")
    
    if selected_view == "🤝 Synergies":
        render_section_header("Synergy Analysis", "Deal Mechanics")
        
        # Get financials safely
        target_revenue = m2.get("revenue", 0)
        if target_revenue <= 0:
            target_revenue = 100e9  # Fallback for demo
            render_banner("warning", "Estimate Used", "Target revenue not available, using estimate.")
        
        target_cogs = target_revenue * 0.55  # COGS estimate
        target_opex = target_revenue * 0.20  # Operating expenses
        total_costs = target_cogs + target_opex
        
        col1, col2 = st.columns(2)
        with col1:
            rev_syn_pct = st.slider("Revenue Synergy (% of Target Revenue)", 0, 30, 10, key="rev_syn", on_change=invalidate_cached_valuation)
            ramp_years = st.selectbox("Full Realization Period (Years)", [1, 2, 3, 4], index=1, key="ramp", on_change=invalidate_cached_valuation)
        with col2:
            cost_syn_pct = st.slider("Cost Synergy (% of Operating Costs)", 0, 40, 15, key="cost_syn", on_change=invalidate_cached_valuation)
            synergy_prob = st.slider("Probability of Achievement", 0.5, 1.0, 0.75, key="syn_prob", on_change=invalidate_cached_valuation)
        
        # Calculate synergies
        annual_rev_synergy = target_revenue * (rev_syn_pct / 100)
        annual_cost_synergy = total_costs * (cost_syn_pct / 100)
        
        # Use addition, not subtraction
        total_annual_synergy = annual_rev_synergy + annual_cost_synergy
        risk_adjusted_synergy = total_annual_synergy * synergy_prob
        
        # PV with ramp-up
        wacc = dcf2.wacc if dcf2.wacc > 0 else 0.12
        pv_synergy = 0
        
        for year in range(1, ramp_years + 1):
            realized_pct = year / ramp_years
            year_synergy = risk_adjusted_synergy * realized_pct
            discount_factor = 1 / (1 + wacc) ** year
            pv_synergy += year_synergy * discount_factor
        
        # Terminal synergy value (after ramp-up period)
        if wacc > dcf2.terminal_growth:
            terminal_synergy = (risk_adjusted_synergy / (wacc - dcf2.terminal_growth)) / (1 + wacc) ** ramp_years
        else:
            terminal_synergy = risk_adjusted_synergy * 10 / (1 + wacc) ** ramp_years
        
        total_pv_synergy = pv_synergy + terminal_synergy
        
        # ============================================================
        # FIX 1: Synergy cap to prevent formula blowup
        # ============================================================
        MAX_SYNERGY_MULTIPLE = 15
        total_pv_synergy = min(total_pv_synergy, risk_adjusted_synergy * MAX_SYNERGY_MULTIPLE)
        
        # Ensure no negative values
        total_pv_synergy = max(total_pv_synergy, 0)
        risk_adjusted_synergy = max(risk_adjusted_synergy, 0)

        def get_synergy_values():
            """Capture current synergy values for report"""
            annual_total = annual_rev_synergy + annual_cost_synergy
            if annual_total > 0:
                rev_share = annual_rev_synergy / annual_total
                cost_share = annual_cost_synergy / annual_total
                pv_rev = total_pv_synergy * rev_share
                pv_cost = total_pv_synergy * cost_share
            else:
                pv_rev = 0
                pv_cost = 0
            return {
                "annual_rev": annual_rev_synergy,
                "annual_cost": annual_cost_synergy,
                "annual_total": annual_total,
                "pv_total": total_pv_synergy,
                "pv_rev": pv_rev,
                "pv_cost": pv_cost,
            }

        # Update synergy values in session state
        st.session_state.synergy_values = get_synergy_values()
        
        # Recompute accretion data if synergy values changed
        if st.session_state.accretion_data:
            ppa_settings = {
                "intangibles_write_up_pct": st.session_state.ppa_intangibles_write_up_pct,
                "useful_life_years": st.session_state.ppa_useful_life_years,
                "interest_rate_pct": st.session_state.ppa_interest_rate_pct,
            }
            updated_accretion = compute_accretion_data(
                m1, m2, offer_premium, cash_pct, total_annual_synergy, tax_rate, ppa_settings
            )
            st.session_state.accretion_data = updated_accretion
        
        render_metric_grid([
            {"label": "Annual Revenue Synergy", "value": f"₹{annual_rev_synergy/1e9:,.2f}B"},
            {"label": "Annual Cost Synergy", "value": f"₹{annual_cost_synergy/1e9:,.2f}B"},
            {"label": "PV of Synergies", "value": f"₹{total_pv_synergy/1e9:,.2f}B"},
        ], columns=3)
        
        # Show warning if synergies seem unrealistic
        if total_pv_synergy > target_revenue * 2:
            render_banner("warning", "Synergy Check", "Synergies seem unusually high - please review assumptions.")
        elif total_pv_synergy < 0:
            render_banner("error", "Synergy Error", "Negative synergy calculation - this indicates a bug.")
        
        st.caption(f"Including {synergy_prob*100:.0f}% probability and {ramp_years}-year ramp-up")
    
    if selected_view == "💰 Deal Mechanics":
        render_section_header("Deal Mechanics", "Accretion / Dilution")

        if st.session_state.assumptions.has_ai_assumptions():
            render_banner("info", "AI Active", "Control premium and multiples come from AI analysis.")
        
        # Use pre-computed accretion data from session state
        acc_data = st.session_state.accretion_data
        if acc_data:
            render_metric_grid([
                {"label": "Offer Value", "value": f"₹{acc_data['offer_value']/1e9:.1f}B"},
                {"label": "Premium to Market", "value": f"{offer_premium:.0f}%"},
                {"label": "Acquirer EPS (Standalone)", "value": f"₹{acc_data['acquirer_eps']:,.2f}"},
                {"label": "Pro-Forma EPS", "value": f"₹{acc_data['pro_forma_eps']:,.2f}"},
                {
                    "label": "Accretion / Dilution",
                    "value": f"{acc_data['accretion']:+.1f}%",
                    "delta": "▲ Positive" if acc_data['accretion'] > 0 else "▼ Negative",
                    "delta_state": "positive" if acc_data['accretion'] > 0 else "negative",
                },
                {"label": "PPA + Financing Drag (After Tax)", "value": f"₹{acc_data.get('ppa_drag_after_tax', 0)/1e9:.2f}B"},
            ], columns=3)
            
            if acc_data['accretion'] > 5:
                render_banner("success", "Accretive", f"Deal is {acc_data['accretion']:.1f}% accretive - strong value creation.")
            elif acc_data['accretion'] > 0:
                render_banner("info", "Accretive", f"Deal is {acc_data['accretion']:.1f}% accretive - modest value creation.")
            else:
                render_banner("warning", "Dilutive", f"Deal is {acc_data['accretion']:.1f}% dilutive - requires strategic justification.")

            if st.session_state.assumptions.has_ai_assumptions():
                ai_rationales = st.session_state.assumptions.get_ai_rationales()
                if "control_premium" in ai_rationales:
                    st.caption(f"🤖 AI Rationale for Control Premium: {ai_rationales['control_premium'][:200]}")
        else:
            render_banner("error", "Accretion Data Missing", "Please rerun analysis.")
    
    if selected_view == "🏗️ PPA & Pro Forma":
        render_section_header("Purchase Price Allocation (PPA) & Pro Forma Balance Sheet", "PPA")

        # Get balance sheet data for both companies
        acquirer_bs = company1.get_balance_sheet_metrics() if hasattr(company1, 'get_balance_sheet_metrics') else {}
        target_bs = company2.get_balance_sheet_metrics() if hasattr(company2, 'get_balance_sheet_metrics') else {}

        # Get key metrics
        acquirer_metrics = company1.get_key_metrics()
        target_metrics = company2.get_key_metrics()

        # Calculate deal value
        target_market_cap = target_metrics.get("market_cap", 100e9)
        offer_value = target_market_cap * (1 + offer_premium / 100)

        # Get balance sheet values or estimates
        target_book_value = target_bs.get("book_value")
        if not target_book_value or target_book_value <= 0:
            # Estimate from market cap and P/B ratio
            pb_ratio = target_metrics.get("pb_ratio")
            if pb_ratio and pb_ratio > 0:
                target_book_value = target_market_cap / pb_ratio
            else:
                # Assume 2x book value typical for Indian companies
                target_book_value = target_market_cap / 2

        target_tangible_book = target_bs.get("tangible_book_value", target_book_value * 0.7)

        # PPA assumptions - user adjustable
        st.markdown("### Purchase Price Allocation Assumptions")
        col1, col2, col3 = st.columns(3)
        with col1:
            ppe_write_up_pct = st.number_input(
                "PP&E Write-up (% of Purchase Price)",
                min_value=0.0, max_value=30.0, step=1.0,
                help="Fair value adjustment for property, plant & equipment",
                key="ppa_ppe_write_up_pct",
                on_change=invalidate_cached_valuation,
            ) / 100

        with col2:
            intangibles_write_up_pct = st.number_input(
                "Intangibles Write-up (% of Purchase Price)",
                min_value=0.0, max_value=40.0, step=1.0,
                help="Identifiable intangibles (brands, customer relationships, technology)",
                key="ppa_intangibles_write_up_pct",
                on_change=invalidate_cached_valuation,
            ) / 100

        with col3:
            transaction_fees_pct = st.number_input(
                "Transaction Fees (% of Deal Value)",
                min_value=0.0, max_value=5.0, step=0.25,
                help="Legal, advisory, and banking fees",
                key="ppa_transaction_fees_pct",
                on_change=invalidate_cached_valuation,
            ) / 100

        # Run PPA calculation
        ppa_result = calculate_goodwill_and_ppa(
            purchase_price=offer_value,
            target_book_value=target_book_value,
            target_tangible_book_value=target_tangible_book,
            ppe_write_up_pct=ppe_write_up_pct,
            intangibles_write_up_pct=intangibles_write_up_pct
        )

        # Display PPA Summary
        st.markdown("### 💰 Purchase Price Allocation Summary")
        goodwill_to_price = (ppa_result['goodwill'] / offer_value * 100) if offer_value > 0 else 0
        render_metric_grid([
            {"label": "Purchase Price", "value": f"₹{offer_value/1e9:.1f}B"},
            {"label": "Target Book Value", "value": f"₹{target_book_value/1e9:.1f}B"},
            {"label": "Goodwill Created", "value": f"₹{ppa_result['goodwill']/1e9:.1f}B"},
            {"label": "Goodwill / Purchase Price", "value": f"{goodwill_to_price:.1f}%"},
        ], columns=4)

        # PPA Allocation Table
        st.markdown("#### PPA Allocation Breakdown")
        ppa_table = create_ppa_summary_table(ppa_result)
        render_table(ppa_table)

        # Show warning if goodwill is too high
        if ppa_result['goodwill'] / offer_value > 0.5:
            render_banner("warning", "Goodwill Risk", "Goodwill exceeds 50% of purchase price - high risk of future impairment.")
        elif ppa_result['goodwill'] < 0:
            render_banner("error", "Negative Goodwill", "This is a bargain purchase. Consider adjusting write-up assumptions.")

        # ========== SOURCES & USES TABLE ==========
        st.markdown("---")
        st.markdown("### 📊 Sources & Uses of Funds")

        # Calculate financing
        cash_pct_decimal = cash_pct / 100
        stock_pct_decimal = 1 - cash_pct_decimal

        acquirer_cash = acquirer_bs.get("cash", acquirer_metrics.get("cash", 0))
        target_debt = target_bs.get("total_debt", target_metrics.get("total_debt", 0))

        cash_portion = offer_value * cash_pct_decimal
        stock_portion = offer_value * stock_pct_decimal

        # Assume acquirer uses 40% of available cash
        available_acquirer_cash = acquirer_cash * 0.4
        cash_from_acquirer = min(cash_portion, available_acquirer_cash)
        new_debt_needed = max(0, cash_portion - cash_from_acquirer)

        financing = {
            "cash_from_acquirer": cash_from_acquirer,
            "new_debt": new_debt_needed,
            "stock_consideration": stock_portion
        }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Sources")
            sources_df, _ = create_sources_uses_table(
                offer_value, cash_pct, stock_pct_decimal * 100,
                acquirer_cash, target_debt, transaction_fees_pct, 0.005
            )
            render_table(sources_df)

            # Format as currency
            st.markdown("**Financing Mix Visualization**")
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels=sources_df['Source'][:3].tolist(),
                values=sources_df['Amount (Rs.B)'][:3].tolist(),
                hole=0.4,
                marker_colors=['#00B4FF', '#34D399', '#FBBF24']
            )])
            fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                        "autoScale2d", "resetScale2d", "hoverClosestCartesian", "hoverCompareCartesian",
                        "toggleSpikelines"
                    ],
                },
            )

        with col2:
            st.markdown("#### Uses")
            _, uses_df = create_sources_uses_table(
                offer_value, cash_pct, stock_pct_decimal * 100,
                acquirer_cash, target_debt, transaction_fees_pct, 0.005
            )
            render_table(uses_df)

        # ========== PRO FORMA BALANCE SHEET ==========
        st.markdown("---")
        st.markdown("### 📋 Pro Forma Combined Balance Sheet")

        with st.expander("Pro Forma Balance Sheet Details", expanded=True):
            pro_forma_df = create_pro_forma_balance_sheet(
                acquirer_bs, target_bs, ppa_result, financing,
                st.session_state.assumptions.tax_rate
            )

            # Format the dataframe for better display
            render_table(pro_forma_df.round(1))

        # Key pro forma metrics
        st.markdown("#### Key Pro Forma Metrics")

        # Extract combined totals
        combined_row = pro_forma_df[pro_forma_df['Item'] == 'Total Assets']
        if not combined_row.empty:
            combined_assets = combined_row['Pro Forma Combined (Rs.B)'].values[0]

            new_leverage = (financing['new_debt'] + target_debt) / (acquirer_metrics.get('market_cap', 1) + stock_portion) * 100
            render_metric_grid([
                {"label": "Combined Assets", "value": f"₹{combined_assets:.1f}B"},
                {"label": "Pro Forma Leverage", "value": f"{new_leverage:.1f}%"},
                {"label": "New Goodwill", "value": f"₹{ppa_result['goodwill']/1e9:.1f}B"},
                {"label": "New Shares Issued", "value": f"{financing['stock_consideration'] / acquirer_metrics.get('current_price', 1000):,.0f}M"},
            ], columns=4)

        # Amortization impact
        st.markdown("---")
        st.markdown("### 📉 PPA Amortization Impact")

        # Calculate annual amortization (typically 5-20 years for intangibles)
        useful_life_years = st.slider(
            "Intangible Asset Useful Life (Years)",
            min_value=3, max_value=20,
            help="Period over which identifiable intangibles are amortized",
            key="ppa_useful_life_years",
            on_change=invalidate_cached_valuation,
        )

        annual_amortization = ppa_result['intangibles_write_up'] / useful_life_years
        annual_amortization_after_tax = annual_amortization * (1 - st.session_state.assumptions.tax_rate)

        render_metric_grid([
            {"label": "Annual Amortization Expense", "value": f"₹{annual_amortization/1e9:.2f}B"},
            {"label": "After-Tax EPS Impact", "value": f"₹{annual_amortization_after_tax / accretion_data.get('total_shares', 1e9):.2f} per share"},
        ], columns=2)
        st.caption(f"Over {useful_life_years} years (straight-line)")
        st.caption("Reduces pro-forma earnings")

        # Financing terms
        st.markdown("---")
        st.markdown("### 💳 Financing Terms")

        col1, col2 = st.columns(2)
        with col1:
            interest_rate = st.number_input(
                "New Debt Interest Rate (%)",
                min_value=5.0, max_value=15.0, step=0.5,
                key="ppa_interest_rate_pct",
                on_change=invalidate_cached_valuation,
            ) / 100

        with col2:
            debt_tenure_years = st.number_input(
                "Debt Tenure (Years)",
                min_value=1, max_value=10,
                key="ppa_debt_tenure_years",
                on_change=invalidate_cached_valuation,
            )

        annual_interest = financing['new_debt'] * interest_rate
        annual_interest_after_tax = annual_interest * (1 - st.session_state.assumptions.tax_rate)

        render_metric_grid([
            {"label": "New Debt Principal", "value": f"₹{financing['new_debt']/1e9:.1f}B"},
            {"label": "Annual Interest Expense", "value": f"₹{annual_interest/1e9:.2f}B"},
            {"label": "After-Tax Interest", "value": f"₹{annual_interest_after_tax/1e9:.2f}B"},
        ], columns=3)

        # Summary impact on earnings
        st.markdown("---")
        st.markdown("### 📊 Total Impact on Pro Forma Earnings")

        total_annual_impact = -annual_amortization_after_tax - annual_interest_after_tax
        impact_per_share = total_annual_impact / accretion_data.get('total_shares', 1e9)

        render_metric_grid([
            {
                "label": "Total Annual PPA Impact",
                "value": f"₹{total_annual_impact/1e9:.2f}B",
                "delta": f"-{abs(total_annual_impact/1e9):.2f}B",
                "delta_state": "negative",
            },
            {
                "label": "Impact on Pro Forma EPS",
                "value": f"₹{impact_per_share:.2f}",
                "delta": f"-{abs(impact_per_share):.2f}",
                "delta_state": "negative",
            },
        ], columns=2)

        render_banner("info", "Note", "PPA amortization and additional interest expense reduce pro-forma earnings. Synergies must overcome these headwinds for accretion.")

    if selected_view == "📝 Memo":
        render_section_header("Investment Memo", "Memo")

        if st.session_state.assumptions.has_ai_assumptions():
            render_banner("info", "AI Active", "Recommendation incorporates AI-driven valuation inputs.")
        
        # Generate recommendation using pre-computed data and DCF implied premium
        acc_data = st.session_state.accretion_data
        if acc_data:
            accretion_val = acc_data['accretion']
        else:
            accretion_val = 0
        
        # ============================================================
        # FIX 2: Improved recommendation logic that catches overvaluation
        # ============================================================
        # Base recommendation logic
        if accretion_val > 5 and dcf2.implied_premium > -20:
            recommendation = "PROCEED"
            rec_color = "success"
        elif accretion_val > 0 or dcf2.implied_premium > -30:
            recommendation = "CONSIDER WITH CAUTION"
            rec_color = "warning"
        else:
            recommendation = "DO NOT PROCEED"
            rec_color = "error"
        
        # Override: if DCF shows significant overvaluation despite accretion, downgrade
        # This catches exactly the Reliance/IndiGo situation where accretion looks good
        # but the underlying asset is overpriced per DCF
        if dcf2.implied_premium < -15 and accretion_val < 10:
            recommendation = "CONSIDER WITH CAUTION"
            rec_color = "warning"
        
        verdict_icon = "PROCEED" if rec_color == "success" else "CAUTION" if rec_color == "warning" else "DO NOT PROCEED"
        verdict_color = THEME["positive"] if rec_color == "success" else THEME["warning"] if rec_color == "warning" else THEME["negative"]
        st.markdown(
            f"""
            <div class='banner {rec_color}' style='padding:20px 22px;'>
                <div class='banner-title'>Recommendation</div>
                <div style='font-size:48px; line-height:1; font-weight:800; color:{verdict_color}; margin-bottom:8px;'>{escape(verdict_icon)}</div>
                <div style='font-size:1rem; color:inherit;'>{escape(recommendation)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        render_section_header("Key Drivers", "Drivers")
        
        fairness_text = dcf2.fairness_rating.lower() if hasattr(dcf2, 'fairness_rating') else "fairly valued"
        
        st.markdown(
            f"""
            <ul style='margin:0 0 0 18px; color:{THEME['text_secondary']}; line-height:1.8;'>
                <li><strong style='color:{THEME['text']};'>Valuation:</strong> Target is <strong>{fairness_text}</strong> vs current price ({dcf2.implied_premium:+.0f}% premium)</li>
                <li><strong style='color:{THEME['text']};'>Accretion:</strong> Deal is <strong>{accretion_val:+.1f}%</strong> to Acquirer EPS</li>
                <li><strong style='color:{THEME['text']};'>Synergies:</strong> Identified Rs.{st.session_state.synergy_values.get('pv_total', 0)/1e9:.1f}B in PV synergies</li>
                <li><strong style='color:{THEME['text']};'>Multiple:</strong> Offer represents {offer_premium:.0f}% premium to current price</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.assumptions.has_ai_assumptions():
            render_section_header("AI Contribution to Analysis", "AI")
            ai_rationales = st.session_state.assumptions.get_ai_rationales()
            st.markdown(
                f"""
                - **Stage 1 Growth:** {st.session_state.assumptions.stage1_growth:.1%} ({ai_rationales.get('stage1_growth', 'AI-estimated based on sector and historicals')[:150]})
                - **Terminal Growth:** {st.session_state.assumptions.terminal_growth:.1%} ({ai_rationales.get('terminal_growth', 'Long-term GDP-linked estimate')[:150]})
                - **EV/EBITDA Multiple:** {st.session_state.assumptions.ev_ebitda_multiple:.1f}x ({ai_rationales.get('ev_ebitda_multiple', 'Sector benchmark')[:150]})
                """
            )
        
        # Add warning if DCF shows overvaluation despite accretion
        if dcf2.implied_premium < -15 and accretion_val > 0:
            render_banner("warning", "Tension Alert", "The deal is accretive but the DCF suggests the target is significantly overvalued. Consider whether strategic rationale justifies paying above intrinsic value.")
        
        render_section_header("Key Risks", "Risk Matrix")
        risks = pd.DataFrame({
            "Risk": ["Integration Execution", "Synergy Realization", "Regulatory", "Market Reaction"],
            "Severity": ["HIGH", "MEDIUM", "MEDIUM", "LOW"],
            "Mitigation": [
                "Phased integration with dedicated team",
                "Conservative targets with contingency",
                "Early regulatory engagement",
                "Clear strategic communication"
            ]
        })
        render_table(risks)
        
        # Export option
        if st.button("📄 Export Analysis (CSV)"):
            render_banner("info", "Export", "Export functionality coming soon.")

    if selected_view == "📌 Assumptions":
        render_section_header("Valuation Assumptions", "AI Intelligence Card")
        company1 = st.session_state.get("loaded_company1")
        company2 = st.session_state.get("loaded_company2")
        sector2 = st.session_state.get("loaded_sector2", "general")

        action_col, _ = st.columns([1, 5])
        with action_col:
            ai_button_clicked = st.button(
                "🤖 Use AI Assumptions",
                type="primary",
                use_container_width=True,
                disabled=not (company1 and company2),
            )

        if ai_button_clicked:
            try:
                api_key = os.getenv("GROQ_API_KEY", "").strip()
                if not api_key:
                    render_banner("error", "Missing API Key", "GROQ_API_KEY not found in environment variables.")
                else:
                    generator = AIAssumptionGenerator(api_key=api_key)
                    result = generator.generate_assumptions(
                        company=company2,
                        acquirer=company1,
                        sector=sector2,
                        user_mode=st.session_state.get("ai_mode", "base"),
                        temperature=0.3,
                    )

                    if result.success:
                        temp_result = AIAssumptionResult(
                            assumptions=result.assumptions,
                            rationales=result.rationales,
                            confidence=result.confidence,
                            key_risks=result.key_risks,
                            raw_response=result.raw_response,
                            success=True,
                            error="",
                        )
                        st.session_state.assumptions = apply_ai_assumptions_to_model(
                            st.session_state.assumptions,
                            temp_result,
                        )
                        st.session_state.ai_generated_assumptions = result.assumptions
                        st.session_state.ai_rationales = result.rationales
                        st.session_state.ai_confidence = result.confidence
                        st.session_state.ai_key_risks = result.key_risks
                        st.session_state.ai_raw_response = result.raw_response
                        st.session_state.ai_assumptions_active = True
                        st.session_state.assumptions_widget_version += 1
                        st.session_state.valuation_results = None
                        st.session_state.last_assumptions_hash = None
                        render_banner("success", "AI Assumptions Applied", f"Confidence: {escape(str(result.confidence))}")
                        st.rerun()
                    else:
                        render_banner("error", "AI Generation Failed", escape(str(result.error)))
            except Exception as exc:
                render_banner("error", "AI Error", escape(str(exc)))

        if not (company1 and company2):
            render_banner("info", "Setup", "Run analysis first to enable the AI assumptions button.")

        dashboard = AssumptionsDashboard(st.session_state.assumptions)
        st.session_state.assumptions = dashboard.render()

        st.divider()
        render_section_header("Active Assumptions Summary", "Summary")

        active_assumptions_df = st.session_state.assumptions.to_dataframe()
        render_table(active_assumptions_df)

        if st.session_state.assumptions.has_ai_assumptions():
            render_banner("success", "AI Active", "AI assumptions are currently active - values above reflect AI-generated inputs.")
        else:
            render_banner("info", "Default Assumptions", "Use the AI generator above to enhance your analysis.")

        assumptions_hash = hash(json.dumps(asdict(st.session_state.assumptions), sort_keys=True))
        if st.session_state.last_assumptions_hash != assumptions_hash:
            st.session_state.last_assumptions_hash = assumptions_hash
            st.session_state.valuation_results = None

else:
    # Welcome screen
    render_banner("info", "Welcome", "Configure companies in sidebar and click Run Analysis.")
    
    st.markdown("""
    ### Features
    
    - **3-Stage DCF** with proper FCF calculation and mid-year discounting
    - **Trading Comps** with sector-appropriate multiples
    - **Precedent Transactions** with control premium analysis
    - **Football Field** visualization of all methods
    - **Sensitivity Analysis** with interactive heatmaps
    - **Synergy Valuation** with ramp-up and probability
    - **Accretion/Dilution** analysis
    - **Sector detection** for appropriate assumptions
    
    ### Example Companies
    
    | Company | Ticker | Sector |
    |---------|--------|--------|
    | Reliance Industries | RELIANCE.NS | Energy |
    | Infosys | INFY.NS | Technology |
    | HDFC Bank | HDFCBANK.NS | Financial |
    | ITC | ITC.NS | Consumer |
    """)