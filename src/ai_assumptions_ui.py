"""Dedicated AI Assumptions UI component integrated into the Assumptions tab."""

from __future__ import annotations

import os
from contextlib import contextmanager
from html import escape
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.ai_assumptions import (
    AIAssumptionGenerator,
    AIAssumptionResult,
    apply_ai_assumptions_to_model,
)
from src.assumptions import ValuationAssumptions


def _safe_pct(value: Any) -> str:
    """Format percentage-like values safely for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        display_value = float(value)
        if abs(display_value) > 1:
            display_value /= 100
        return f"{display_value:.1%}"
    return str(value)


def _safe_mult(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}x"
    return str(value)


def _get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(data, dict) and key in data and data[key] is not None:
            return data[key]
    return default


@contextmanager
def _loading(message: str):
    placeholder = st.empty()
    placeholder.markdown(
        f"<div class='loading-strip'></div><div class='loading-message'>{escape(message)}</div>",
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def _banner(kind: str, title: str, message: str):
    st.markdown(
        f"""
        <div class='banner {kind}'>
            <div class='banner-title'>{escape(title)}</div>
            <div class='banner-body'>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_assumptions_ui(
    company,
    acquirer,
    sector: str,
    assumptions: ValuationAssumptions,
) -> ValuationAssumptions:
    """Render the AI assumptions UI and return the active assumptions object."""

    st.markdown("### 🤖 AI-Powered Assumption Generator")
    st.markdown("Generate intelligent, sector-specific valuation assumptions using Groq LLM.")

    if assumptions.has_ai_assumptions():
        _banner("success", "AI Active", "Currently using AI-generated values.")
    else:
        _banner("info", "Default Assumptions", "Generate AI assumptions to override the model.")

    st.divider()

    if company is None or acquirer is None:
        _banner("warning", "Context Required", "Run analysis first to load company context for AI assumptions.")
        return assumptions

    mode_options = ["conservative", "base", "optimistic", "strategic"]
    current_mode = st.session_state.get("ai_mode", "base")
    if current_mode not in mode_options:
        current_mode = "base"

    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Assumption Mode",
            mode_options,
            index=mode_options.index(current_mode),
            help=(
                "Conservative: lower estimates | Base: realistic mid-range | "
                "Optimistic: higher growth | Strategic: higher synergies"
            ),
        )

    with col2:
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.1,
            max_value=0.7,
            value=0.3,
            step=0.05,
            help="Lower = more deterministic, higher = more creative",
        )

    generate_clicked = st.button(
        "🎯 Generate AI Assumptions",
        type="primary",
        use_container_width=True,
        disabled=company is None or acquirer is None,
    )

    if generate_clicked:
        with _loading("AI is analyzing sector context and generating assumptions..."):
            try:
                api_key = os.getenv("GROQ_API_KEY", "").strip()
                if not api_key:
                    _banner("error", "Missing API Key", "GROQ_API_KEY not found in environment variables.")
                    _banner("info", "Setup", "Add GROQ_API_KEY to your .env file and reload the app.")
                else:
                    generator = AIAssumptionGenerator(api_key=api_key)
                    result = generator.generate_assumptions(
                        company=company,
                        acquirer=acquirer,
                        sector=sector,
                        user_mode=mode,
                        temperature=temperature,
                    )

                    if result.success:
                        st.session_state.ai_generated_assumptions = result.assumptions
                        st.session_state.ai_rationales = result.rationales
                        st.session_state.ai_confidence = result.confidence
                        st.session_state.ai_key_risks = result.key_risks
                        st.session_state.ai_mode = mode
                        st.session_state.ai_raw_response = result.raw_response
                        st.session_state.ai_assumptions_active = False
                        _banner("success", "AI Generated", f"Confidence: {escape(str(result.confidence))}")
                        st.rerun()
                    else:
                        _banner("error", "AI Generation Failed", escape(str(result.error)))
            except Exception as exc:
                _banner("error", "Error", escape(str(exc)))

    if st.session_state.get("ai_generated_assumptions"):
        st.divider()
        st.markdown("### 📊 AI Assumptions Preview")
        st.caption("Review the current values against the AI-generated suggestions before applying them.")

        ai_assumptions = st.session_state.ai_generated_assumptions
        rationales = st.session_state.get("ai_rationales") or {}

        current_values = {
            "stage1_growth": assumptions.stage1_growth,
            "terminal_growth": assumptions.terminal_growth,
            "stage1_ebitda_margin": assumptions.stage1_ebitda_margin,
            "terminal_margin": assumptions.terminal_margin,
            "reinvestment_rate": assumptions.reinvestment_rate,
            "ev_ebitda_multiple": assumptions.ev_ebitda_multiple,
            "pe_multiple": assumptions.pe_multiple,
            "control_premium": assumptions.control_premium,
            "risk_free_rate": assumptions.risk_free_rate,
            "equity_risk_premium": assumptions.equity_risk_premium,
            "tax_rate": assumptions.tax_rate,
            "transaction_ev_ebitda": assumptions.transaction_ev_ebitda,
        }

        field_specs = [
            ("Stage 1 Growth", "stage1_growth", ("dcf", "stage1_growth"), "pct"),
            ("Terminal Growth", "terminal_growth", ("dcf", "terminal_growth"), "pct"),
            ("Stage 1 EBITDA Margin", "stage1_ebitda_margin", ("dcf", "stage1_ebitda_margin"), "pct"),
            ("Terminal Margin", "terminal_margin", ("dcf", "terminal_margin"), "pct"),
            ("Reinvestment Rate", "reinvestment_rate", ("dcf", "reinvestment_rate"), "pct"),
            ("EV/EBITDA Multiple", "ev_ebitda_multiple", ("trading_comps", "ev_ebitda"), "mult"),
            ("P/E Multiple", "pe_multiple", ("trading_comps", "pe_ratio"), "mult"),
            ("Control Premium", "control_premium", ("precedent", "control_premium"), "pct"),
            ("Risk-Free Rate", "risk_free_rate", ("macro", "risk_free_rate"), "pct"),
            ("Equity Risk Premium", "equity_risk_premium", ("macro", "equity_risk_premium"), "pct"),
            ("Tax Rate", "tax_rate", ("macro", "tax_rate"), "pct"),
            ("Transaction EV/EBITDA", "transaction_ev_ebitda", ("precedent", "transaction_ev_ebitda"), "mult"),
        ]

        comparison_rows = []
        for label, current_key, (section, ai_key), value_type in field_specs:
            if not isinstance(ai_assumptions, dict):
                section_data = {}
            elif section == "trading_comps":
                section_data = ai_assumptions.get("trading_comps", ai_assumptions.get("comps", {}))
            else:
                section_data = ai_assumptions.get(section, {})
            ai_value = _get(section_data, ai_key)
            if ai_value is None:
                continue

            current_value = current_values.get(current_key)
            if value_type == "mult":
                current_str = _safe_mult(current_value)
                ai_str = _safe_mult(ai_value)
                if isinstance(current_value, (int, float)) and isinstance(ai_value, (int, float)):
                    change_str = f"{ai_value - current_value:+.1f}x"
                else:
                    change_str = ""
            else:
                current_str = _safe_pct(current_value)
                ai_str = _safe_pct(ai_value)
                if isinstance(current_value, (int, float)) and isinstance(ai_value, (int, float)):
                    change_str = f"{ai_value - current_value:+.1%}"
                else:
                    change_str = ""

            rationale = rationales.get(current_key) or rationales.get(ai_key, "")
            comparison_rows.append(
                {
                    "Parameter": label,
                    "Current Value": current_str,
                    "AI Generated": ai_str,
                    "Change": change_str,
                    "Rationale": rationale[:220] + ("..." if len(rationale) > 220 else ""),
                }
            )

        if comparison_rows:
            st.markdown(
                pd.DataFrame(comparison_rows).to_html(index=False, classes="pill-table"),
                unsafe_allow_html=True,
            )

        st.markdown(f"**Confidence Level:** {st.session_state.get('ai_confidence', 'Medium')}")

        if st.session_state.get("ai_key_risks"):
            with st.expander("⚠️ Key Risks Identified by AI", expanded=False):
                for risk in st.session_state.get("ai_key_risks", []):
                    st.write(f"• {risk}")

        col_apply, col_discard = st.columns(2)
        with col_apply:
            if st.button("✅ Use AI Assumptions", type="primary", use_container_width=True):
                temp_result = AIAssumptionResult(
                    assumptions=st.session_state.ai_generated_assumptions,
                    rationales=st.session_state.ai_rationales or {},
                    confidence=st.session_state.ai_confidence or "Medium",
                    key_risks=st.session_state.ai_key_risks or [],
                    raw_response=st.session_state.get("ai_raw_response", ""),
                    success=True,
                    error="",
                )

                updated_assumptions = apply_ai_assumptions_to_model(assumptions, temp_result)
                st.session_state.assumptions = updated_assumptions
                st.session_state.ai_assumptions_active = True
                st.session_state.assumptions_widget_version = int(st.session_state.get("assumptions_widget_version", 0)) + 1
                st.session_state.valuation_results = None
                st.session_state.last_assumptions_hash = None

                _banner("success", "AI Assumptions Applied", "Re-run analysis to update valuation outputs.")
                st.rerun()

        with col_discard:
            if st.button("❌ Discard", use_container_width=True):
                st.session_state.ai_generated_assumptions = None
                st.session_state.ai_rationales = None
                st.session_state.ai_confidence = None
                st.session_state.ai_key_risks = None
                st.session_state.ai_raw_response = ""
                st.rerun()

    if assumptions.has_ai_assumptions():
        st.divider()
        _banner("info", "AI Active", "The valuation model is using AI-generated values.")
        if st.button("🗑️ Clear AI Assumptions (Revert to Defaults)", use_container_width=True):
            st.session_state.assumptions = ValuationAssumptions(
                ticker=company.ticker if company else ""
            )
            st.session_state.ai_assumptions_active = False
            st.session_state.ai_generated_assumptions = None
            st.session_state.ai_rationales = None
            st.session_state.ai_confidence = None
            st.session_state.ai_key_risks = None
            st.session_state.assumptions_widget_version = int(st.session_state.get("assumptions_widget_version", 0)) + 1
            st.session_state.valuation_results = None
            st.session_state.last_assumptions_hash = None
            _banner("success", "Reverted", "Default assumptions restored.")
            st.rerun()

    return st.session_state.assumptions


def render_ai_assumptions_menu(
    company,
    acquirer,
    sector: str,
    assumptions: ValuationAssumptions,
) -> ValuationAssumptions:
    """Compatibility wrapper for existing callers."""

    return render_ai_assumptions_ui(company, acquirer, sector, assumptions)