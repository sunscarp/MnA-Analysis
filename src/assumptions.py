import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf

def get_10y_yield(country: str = "IN") -> float:
    """Get actual 10-year government bond yield."""
    try:
        if country == "IN":
            bond = yf.Ticker("^IN10YGS")
            data = bond.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1]) / 100
        else:
            bond = yf.Ticker("^TNX")
            data = bond.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1]) / 100
    except Exception:
        pass

    return 0.071 if country == "IN" else 0.045


def get_market_params(ticker: str) -> Dict[str, float]:
    """Get market-appropriate parameters based on ticker."""
    ticker_upper = (ticker or "").upper()

    if ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO") or not ticker_upper:
        return {
            "risk_free_rate": get_10y_yield("IN"),
            "equity_risk_premium": 0.055,
            "terminal_growth": 0.065,
            "market": "India"
        }

    return {
        "risk_free_rate": get_10y_yield("US"),
        "equity_risk_premium": 0.05,
        "terminal_growth": 0.025,
        "market": "US"
    }

@dataclass
class ValuationAssumptions:
    """Central assumptions repository - market aware defaults"""

    ticker: str = ""

    def __post_init__(self):
        """Initialize with market-appropriate values."""
        market_params = get_market_params(self.ticker)
        self.risk_free_rate = market_params["risk_free_rate"]
        self.equity_risk_premium = market_params["equity_risk_premium"]
        self.terminal_growth = market_params["terminal_growth"]
        self._market = market_params["market"]

        if self.stage1_growth is None:
            self.stage1_growth = 0.12 if self._market == "India" else 0.08
        if self.stage1_ebitda_margin is None:
            self.stage1_ebitda_margin = 0.18 if self._market == "India" else 0.16
        if self.terminal_margin is None:
            self.terminal_margin = max(self.stage1_ebitda_margin - 0.03, 0.08)
        if self.debt_to_capital is None:
            self.debt_to_capital = 0.30
        if self.ev_revenue_multiple is None:
            self.ev_revenue_multiple = 3.0
        if self.ev_ebitda_multiple is None:
            self.ev_ebitda_multiple = 12.0
        if self.pe_multiple is None:
            self.pe_multiple = 20.0
        if self.pb_multiple is None:
            self.pb_multiple = 2.0
        if self.transaction_ev_revenue is None:
            self.transaction_ev_revenue = 4.0
        if self.transaction_ev_ebitda is None:
            self.transaction_ev_ebitda = 14.0

    # Macro Assumptions
    risk_free_rate: float = None
    equity_risk_premium: float = 0.055
    tax_rate: float = 0.25

    # DCF Assumptions
    stage1_years: int = 3
    stage2_years: int = 4
    stage1_growth: float = None
    terminal_growth: float = None
    stage1_ebitda_margin: float = None
    terminal_margin: float = None
    reinvestment_rate: float = 0.40

    # Capital Structure
    debt_to_capital: float = None
    cost_of_debt_spread: float = 0.02

    # Trading Comps
    ev_revenue_multiple: float = None
    ev_ebitda_multiple: float = None
    pe_multiple: float = None
    pb_multiple: float = None

    # Precedent Transactions
    control_premium: float = 0.25
    transaction_ev_revenue: float = None
    transaction_ev_ebitda: float = None
    
    # Industry-specific adjustments
    sector_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "technology": 1.20,
        "financial": 0.90,
        "industrial": 0.85,
        "consumer": 1.00,
        "healthcare": 1.15,
        "airlines": 0.70  # Added for IndiGo
    })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert assumptions to DataFrame for display"""
        data = {
            "Category": [],
            "Assumption": [],
            "Value": [],
            "Description": []
        }
        
        # Macro
        macro_items = [
            ("Macro", "Risk-Free Rate", f"{self.risk_free_rate:.1%}", "10-year government bond yield"),
            ("Macro", "Equity Risk Premium", f"{self.equity_risk_premium:.1%}", "Market risk premium"),
            ("Macro", "Tax Rate", f"{self.tax_rate:.1%}", "Effective tax rate")
        ]
        
        # DCF
        dcf_items = [
            ("DCF", "Stage 1 Years", self.stage1_years, "High growth period"),
            ("DCF", "Stage 2 Years", self.stage2_years, "Transition period"),
            ("DCF", "Stage 1 Growth", f"{self.stage1_growth:.1%}", "Revenue growth in high growth phase"),
            ("DCF", "Terminal Growth", f"{self.terminal_growth:.1%}", "Long-term stable growth"),
            ("DCF", "Stage 1 EBITDA Margin", f"{self.stage1_ebitda_margin:.1%}", "Margin during high growth"),
            ("DCF", "Reinvestment Rate", f"{self.reinvestment_rate:.1%}", "Capital reinvestment rate")
        ]
        
        # Multiples
        multiples_items = [
            ("Trading Comps", "EV/Revenue", f"{self.ev_revenue_multiple:.1f}x", "Enterprise value to revenue"),
            ("Trading Comps", "EV/EBITDA", f"{self.ev_ebitda_multiple:.1f}x", "Enterprise value to EBITDA"),
            ("Trading Comps", "P/E Ratio", f"{self.pe_multiple:.1f}x", "Price to earnings"),
            ("Precedent", "Control Premium", f"{self.control_premium:.1%}", "Acquisition premium over market"),
            ("Precedent", "Transaction EV/Revenue", f"{self.transaction_ev_revenue:.1f}x", "Deal multiple to revenue")
        ]
        
        all_items = macro_items + dcf_items + multiples_items
        
        for cat, name, val, desc in all_items:
            data["Category"].append(cat)
            data["Assumption"].append(name)
            data["Value"].append(val)
            data["Description"].append(desc)
        
        return pd.DataFrame(data)
    
    def get_dcf_kwargs(self) -> Dict[str, Any]:
        """Get DCF-specific assumptions as kwargs"""
        return {
            "stage1_years": self.stage1_years,
            "stage2_years": self.stage2_years,
            "stage1_growth": self.stage1_growth,
            "terminal_growth": self.terminal_growth,
            "tax_rate": self.tax_rate,
            "reinvestment_rate": self.reinvestment_rate
        }
    
    def get_comps_kwargs(self) -> Dict[str, Any]:
        """Get Trading Comps assumptions"""
        return {
            "ev_revenue_multiple": self.ev_revenue_multiple,
            "ev_ebitda_multiple": self.ev_ebitda_multiple,
            "pe_multiple": self.pe_multiple
        }
    
    def get_precedent_kwargs(self) -> Dict[str, Any]:
        """Get Precedent Transactions assumptions"""
        return {
            "median_ev_revenue": self.transaction_ev_revenue,
            "median_ev_ebitda": self.transaction_ev_ebitda,
            "control_premium": self.control_premium
        }

    def has_ai_assumptions(self) -> bool:
        """Check if AI assumptions are currently active."""
        import streamlit as st

        return st.session_state.get("ai_assumptions_active", False)

    def get_ai_rationales(self) -> dict:
        """Get AI rationales if available."""
        import streamlit as st

        return st.session_state.get("ai_rationales", {})

    def update_from_dict(self, updates: dict):
        """Update assumptions from dictionary (e.g., AI output)."""
        for key, value in updates.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

class AssumptionsDashboard:
    """Interactive assumptions dashboard UI"""
    
    def __init__(self, assumptions: ValuationAssumptions = None):
        self.assumptions = assumptions or ValuationAssumptions()
    
    def render(self):
        """Render the assumptions dashboard in Streamlit"""
        st.markdown("### 🎯 Valuation Assumptions Dashboard")
        st.markdown("Adjust assumptions to see real-time impact on valuation")

        widget_version = int(st.session_state.get("assumptions_widget_version", 0))

        def _widget_key(name: str) -> str:
            return f"assumptions_{widget_version}_{name}"

        def _pct_value(value: float, min_value: float, max_value: float) -> float:
            """Convert stored decimal to percent and clamp to widget bounds."""
            value_pct = value * 100
            if value_pct < min_value:
                return min_value
            if value_pct > max_value:
                return max_value
            return value_pct

        def _int_value(value: Any, min_value: int, max_value: int) -> int:
            """Clamp integer widget values to valid bounds."""
            try:
                int_value = int(value)
            except Exception:
                int_value = min_value
            if int_value < min_value:
                return min_value
            if int_value > max_value:
                return max_value
            return int_value

        def _num_value(value: Any, min_value: float, max_value: float) -> float:
            """Clamp numeric widget values to valid bounds."""
            try:
                num_value = float(value)
            except Exception:
                num_value = min_value
            if num_value < min_value:
                return min_value
            if num_value > max_value:
                return max_value
            return num_value
        
        # Create tabs for assumption categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Macro & DCF", "🏭 Capital Structure", "📊 Market Multiples", "📋 All Assumptions"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Macroeconomic")
                self.assumptions.risk_free_rate = st.number_input(
                    "Risk-Free Rate (%)", 
                    min_value=3.0, max_value=12.0,
                    value=_pct_value(self.assumptions.risk_free_rate, 3.0, 12.0),
                    format="%.2f",
                    key=_widget_key("risk_free_rate")
                ) / 100
                
                self.assumptions.equity_risk_premium = st.number_input(
                    "Equity Risk Premium (%)",
                    min_value=4.0, max_value=10.0,
                    value=_pct_value(self.assumptions.equity_risk_premium, 4.0, 10.0),
                    format="%.2f",
                    key=_widget_key("equity_risk_premium")
                ) / 100
                
                self.assumptions.tax_rate = st.number_input(
                    "Tax Rate (%)",
                    min_value=0.0, max_value=40.0,
                    value=_pct_value(self.assumptions.tax_rate, 0.0, 40.0),
                    format="%.2f",
                    key=_widget_key("tax_rate")
                ) / 100
            
            with col2:
                st.markdown("#### DCF - Growth Stages")
                self.assumptions.stage1_years = st.number_input(
                    "Stage 1 (High Growth) Years",
                    min_value=1, max_value=10,
                    value=_int_value(self.assumptions.stage1_years, 1, 10),
                    key=_widget_key("stage1_years")
                )
                
                self.assumptions.stage2_years = st.number_input(
                    "Stage 2 (Transition) Years",
                    min_value=1, max_value=10,
                    value=_int_value(self.assumptions.stage2_years, 1, 10),
                    key=_widget_key("stage2_years")
                )
                
                self.assumptions.stage1_growth = st.number_input(
                    "Stage 1 Growth Rate (%)",
                    min_value=5.0, max_value=35.0,
                    value=_pct_value(self.assumptions.stage1_growth, 5.0, 35.0),
                    format="%.1f",
                    key=_widget_key("stage1_growth")
                ) / 100

                self.assumptions.terminal_growth = st.number_input(
                    "Terminal Growth Rate (%)",
                    min_value=2.0, max_value=7.0,
                    value=_pct_value(self.assumptions.terminal_growth, 2.0, 7.0),
                    format="%.1f",
                    key=_widget_key("terminal_growth")
                ) / 100
            
            st.markdown("#### DCF - Margins")
            col3, col4 = st.columns(2)
            with col3:
                self.assumptions.stage1_ebitda_margin = st.number_input(
                    "Stage 1 EBITDA Margin (%)",
                    min_value=5.0, max_value=50.0,
                    value=_pct_value(self.assumptions.stage1_ebitda_margin, 5.0, 50.0),
                    format="%.1f",
                    key=_widget_key("stage1_ebitda_margin")
                ) / 100
            with col4:
                self.assumptions.terminal_margin = st.number_input(
                    "Terminal EBITDA Margin (%)",
                    min_value=5.0, max_value=40.0,
                    value=_pct_value(self.assumptions.terminal_margin, 5.0, 40.0),
                    format="%.1f",
                    key=_widget_key("terminal_margin")
                ) / 100
            
            self.assumptions.reinvestment_rate = st.number_input(
                "Reinvestment Rate (%)",
                min_value=10.0, max_value=70.0,
                value=_pct_value(self.assumptions.reinvestment_rate, 10.0, 70.0),
                format="%.1f",
                key=_widget_key("reinvestment_rate")
            ) / 100
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                self.assumptions.debt_to_capital = st.number_input(
                    "Debt to Capital Ratio (%)",
                    min_value=0.0, max_value=70.0,
                    value=_pct_value(self.assumptions.debt_to_capital, 0.0, 70.0),
                    format="%.1f",
                    key=_widget_key("debt_to_capital")
                ) / 100
            with col2:
                self.assumptions.cost_of_debt_spread = st.number_input(
                    "Cost of Debt Spread (%)",
                    min_value=1.0, max_value=6.0,
                    value=_pct_value(self.assumptions.cost_of_debt_spread, 1.0, 6.0),
                    format="%.2f",
                    key=_widget_key("cost_of_debt_spread")
                ) / 100
        
        with tab3:
            st.markdown("#### Trading Comparable Multiples")
            col1, col2 = st.columns(2)
            with col1:
                self.assumptions.ev_revenue_multiple = st.number_input(
                    "EV/Revenue Multiple",
                    min_value=0.5, max_value=15.0,
                    value=_num_value(self.assumptions.ev_revenue_multiple, 0.5, 15.0),
                    step=0.5,
                    key=_widget_key("ev_revenue_multiple")
                )
                
                self.assumptions.ev_ebitda_multiple = st.number_input(
                    "EV/EBITDA Multiple",
                    min_value=2.0, max_value=30.0,
                    value=_num_value(self.assumptions.ev_ebitda_multiple, 2.0, 30.0),
                    step=1.0,
                    key=_widget_key("ev_ebitda_multiple")
                )
            with col2:
                self.assumptions.pe_multiple = st.number_input(
                    "P/E Ratio",
                    min_value=5.0, max_value=50.0,
                    value=_num_value(self.assumptions.pe_multiple, 5.0, 50.0),
                    step=1.0,
                    key=_widget_key("pe_multiple")
                )
            
            st.markdown("#### Precedent Transactions")
            col3, col4 = st.columns(2)
            with col3:
                self.assumptions.control_premium = st.number_input(
                    "Control Premium (%)",
                    min_value=10.0, max_value=60.0,
                    value=_pct_value(self.assumptions.control_premium, 10.0, 60.0),
                    format="%.1f",
                    key=_widget_key("control_premium")
                ) / 100
            with col4:
                self.assumptions.transaction_ev_ebitda = st.number_input(
                    "Transaction EV/EBITDA",
                    min_value=4.0, max_value=25.0,
                    value=_num_value(self.assumptions.transaction_ev_ebitda, 4.0, 25.0),
                    step=1.0,
                    key=_widget_key("transaction_ev_ebitda")
                )
        
        with tab4:
            st.markdown(
                self.assumptions.to_dataframe().to_html(index=False, classes="pill-table"),
                unsafe_allow_html=True,
            )
        
        # Export option
        st.divider()
        col_export, _ = st.columns([1, 3])
        with col_export:
            if st.button("📥 Export Assumptions", use_container_width=True):
                csv = self.assumptions.to_dataframe().to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="valuation_assumptions.csv",
                    mime="text/csv"
                )
        
        return self.assumptions