import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
from src.assumptions import get_market_params
from html import escape


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

@dataclass
class DCFResult:
    """Professional DCF result with full transparency"""
    # Core values
    enterprise_value: float
    equity_value: float
    per_share: float
    
    # Value components
    stage1_pv: float
    stage2_pv: float
    terminal_pv: float
    terminal_value: float
    
    # FCF projections (year by year)
    projection_years: List[int]
    revenue_projections: List[float]
    ebit_projections: List[float]
    fcf_projections: List[float]
    
    # Assumptions used
    wacc: float
    terminal_growth: float
    stage1_growth: float
    stage2_growth_decline: float
    
    # Market context
    current_price: float
    implied_premium: float
    fairness_rating: str  # Changed from 'fairness' to 'fairness_rating'
    
    # Sensitivity analysis
    sensitivity_matrix: Dict[str, Dict[float, List[float]]] = field(default_factory=dict)
    
    @property
    def fairness(self):
        """Backward compatibility property"""
        return self.fairness_rating
    
    def get_value_breakdown(self) -> pd.DataFrame:
        """Get breakdown of value components"""
        return pd.DataFrame({
            "Component": ["Stage 1 (High Growth)", "Stage 2 (Transition)", "Terminal Value"],
            "Present Value (₹B)": [self.stage1_pv/1e9, self.stage2_pv/1e9, self.terminal_pv/1e9],
            "Share of Value": [
                f"{self.stage1_pv/self.enterprise_value*100:.1f}%" if self.enterprise_value > 0 else "0%",
                f"{self.stage2_pv/self.enterprise_value*100:.1f}%" if self.enterprise_value > 0 else "0%",
                f"{self.terminal_pv/self.enterprise_value*100:.1f}%" if self.enterprise_value > 0 else "0%"
            ]
        })


class DCFModel:
    """Professional 3-Stage DCF with proper FCF calculation"""
    
    def __init__(self, company, sector: str = "general"):
        self.company = company
        self.sector = sector
        self.metrics = company.get_key_metrics()
        self._validate_financials()
    
    def _validate_financials(self):
        """Validate financial data with realistic bounds"""
        revenue = self.metrics.get("revenue")
        
        # For Indian large caps, revenue should be in billions
        if revenue is None or revenue <= 0:
            # Try to estimate from market cap (rough)
            market_cap = self.metrics.get("market_cap")
            if market_cap and market_cap > 0:
                # Rough estimate: typical PS ratio for Indian market
                sector_ps_ratios = {
                    "technology": 6.0,
                    "financial": 3.0,
                    "industrial": 2.5,
                    "consumer": 3.5,
                    "energy": 1.2,
                    "general": 2.5
                }
                ps_ratio = sector_ps_ratios.get(self.sector, 2.5)
                revenue = market_cap / ps_ratio
            else:
                raise ValueError(f"No valid revenue data for {self.company.ticker}")
        
        # Check for reasonableness (Reliance ~800B, Infosys ~1,500B)
        if revenue < 1e8 and revenue > 0:  # Less than 100M is suspicious for large cap
            import streamlit as st
            _banner("warning", "Revenue Check", f"Revenue ({revenue:,.0f}) seems low. Check if data is correct.")
        
        self.revenue = revenue
        
        # Get EBITDA or estimate
        ebitda = self.metrics.get("ebitda")
        if ebitda is None or ebitda <= 0:
            ebit = self.metrics.get("ebit")
            if ebit and ebit > 0:
                ebitda = ebit * 1.1
            else:
                # Industry-based margin
                sector_margins = {
                    "technology": 0.25,
                    "financial": 0.30,
                    "industrial": 0.15,
                    "consumer": 0.12,
                    "energy": 0.10,
                    "general": 0.15
                }
                margin = sector_margins.get(self.sector, 0.15)
                ebitda = revenue * margin
        
        self.ebitda = ebitda
        self.ebit = self.metrics.get("ebit", ebitda * 0.9)
        
        # Capex and D&A estimates
        self.capex = self.metrics.get("capex", revenue * 0.05)
        self.depreciation = self.metrics.get("depreciation", ebitda - self.ebit if ebitda > self.ebit else revenue * 0.03)
        
    def calculate_wacc(self,
                       risk_free_rate: float = None,
                       equity_risk_premium: float = None,
                       debt_to_equity: float = None,
                       cost_of_debt: float = None,
                       tax_rate: float = 0.25) -> float:
        """
        Calculate WACC using company and market inputs where available.
        """
        market_params = get_market_params(self.company.ticker)
        if risk_free_rate is None:
            risk_free_rate = market_params["risk_free_rate"]
        if equity_risk_premium is None:
            equity_risk_premium = market_params["equity_risk_premium"]

        beta = self.company.info.get("beta")
        if beta is None or beta <= 0:
            beta = 1.0

        if debt_to_equity is None:
            total_debt = self.metrics.get("total_debt") or 0
            market_cap = self.metrics.get("market_cap") or 0
            if market_cap > 0:
                debt_to_equity = total_debt / market_cap
            else:
                debt_to_equity = 0.30

        cost_of_equity = risk_free_rate + beta * equity_risk_premium

        if cost_of_debt is None:
            interest_expense = self.metrics.get("interest_expense") or 0
            total_debt = self.metrics.get("total_debt") or 0
            if total_debt > 0 and interest_expense > 0:
                cost_of_debt = interest_expense / total_debt
            else:
                cost_of_debt = risk_free_rate + 0.02

        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

        debt_ratio = debt_to_equity / (1 + debt_to_equity)
        equity_ratio = 1 - debt_ratio

        wacc = (cost_of_equity * equity_ratio) + (after_tax_cost_of_debt * debt_ratio)

        return round(wacc, 4)
    
    def calculate_free_cash_flow(self, 
                                 revenue: float,
                                 ebitda_margin: float,
                                 tax_rate: float = 0.25,
                                 revenue_growth: float = 0.10,
                                 reinvestment_rate: float = 0.40) -> Tuple[float, float]:
        """
        Proper FCF = EBIT(1-t) + D&A - Capex - ΔNWC
        Returns (fcf, ebit)
        """
        # Calculate EBIT (using EBITDA - D&A)
        ebitda_val = revenue * ebitda_margin
        # Assume D&A is roughly 30% of EBITDA for most industries
        depreciation_val = ebitda_val * 0.3
        ebit = ebitda_val - depreciation_val
        
        # NOPAT
        nopat = ebit * (1 - tax_rate)
        
        # Add back D&A
        nopat_plus_da = nopat + depreciation_val
        
        # Capex (maintenance + growth)
        # Reinvestment rate governs reinvestment against incremental revenue.
        reinvestment_rate = min(max(reinvestment_rate, 0.10), 0.90)
        maintenance_capex = revenue * 0.03  # Maintenance ~3% of revenue
        incremental_reinvestment = revenue * revenue_growth * reinvestment_rate
        growth_capex = incremental_reinvestment * 0.7
        total_capex = maintenance_capex + growth_capex
        
        # NWC change scales with reinvestment intensity.
        nwc_change = incremental_reinvestment * 0.3
        
        fcf = nopat_plus_da - total_capex - nwc_change
        
        # FCF can't be negative for stable companies, but allow for growth stage
        return max(fcf, revenue * 0.01), ebit  # Minimum 1% of revenue FCF
    
    def project_fcf_3stage(self,
                          stage1_years: int = 3,
                          stage2_years: int = 4,
                          stage1_growth: float = 0.12,
                          terminal_growth: float = 0.03,
                          stage1_ebitda_margin: float = None,
                          terminal_margin: float = None,
                          reinvestment_rate: float = 0.40,
                          tax_rate: float = 0.25) -> Tuple[List[float], List[float], List[float]]:
        """
        Professional 3-stage FCF projection
        Returns: (revenues, ebits, fcfs)
        """
        # Set margins based on sector
        if stage1_ebitda_margin is None:
            sector_margins = {
                "technology": 0.28,
                "financial": 0.35,
                "industrial": 0.18,
                "consumer": 0.15,
                "energy": 0.14,
                "general": 0.18
            }
            stage1_ebitda_margin = sector_margins.get(self.sector, 0.18)
        
        if terminal_margin is None:
            terminal_margin = max(stage1_ebitda_margin - 0.04, 0.10)
        
        total_years = stage1_years + stage2_years
        revenues = []
        ebits = []
        fcfs = []
        
        current_revenue = self.revenue
        current_margin = stage1_ebitda_margin
        
        # Annual growth decline for stage 2
        growth_decline = (stage1_growth - terminal_growth) / stage2_years if stage2_years > 0 else 0
        
        # Stage 1: High Growth
        for year in range(1, stage1_years + 1):
            current_revenue *= (1 + stage1_growth)
            revenues.append(current_revenue)
            
            fcf, ebit = self.calculate_free_cash_flow(
                current_revenue, current_margin, tax_rate, stage1_growth, reinvestment_rate
            )
            fcfs.append(fcf)
            ebits.append(ebit)
        
        # Stage 2: Transition
        for year in range(1, stage2_years + 1):
            # Linear growth decline
            growth_rate = stage1_growth - (year * growth_decline)
            growth_rate = max(growth_rate, terminal_growth)
            current_revenue *= (1 + growth_rate)
            revenues.append(current_revenue)
            
            # Linear margin convergence
            margin_step = (stage1_ebitda_margin - terminal_margin) / stage2_years
            current_margin -= margin_step
            current_margin = max(current_margin, terminal_margin)
            
            fcf, ebit = self.calculate_free_cash_flow(
                current_revenue, current_margin, tax_rate, growth_rate, reinvestment_rate
            )
            fcfs.append(fcf)
            ebits.append(ebit)
        
        return revenues, ebits, fcfs
    
    def _run_sensitivity_analysis(self, base_wacc, base_term_growth, base_per_share):
        """Run sensitivity on WACC and Terminal Growth"""
        wacc_vars = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, base_wacc + 0.005, base_wacc + 0.01]
        growth_vars = [base_term_growth - 0.005, base_term_growth, base_term_growth + 0.005]
        
        sensitivity = {
            "wacc_variations": wacc_vars,
            "growth_variations": growth_vars,
            "values": {}
        }
        
        for w in wacc_vars:
            sensitivity["values"][w] = {}
            for g in growth_vars:
                # Approximate recalculation
                if w > 0:
                    value_factor = (base_wacc / w) * ((1 + g) / (1 + base_term_growth))
                    sensitivity["values"][w][g] = base_per_share * value_factor
                else:
                    sensitivity["values"][w][g] = base_per_share
        
        return sensitivity
    
    def run_dcf(self,
                stage1_years: int = 3,
                stage2_years: int = 4,
                stage1_growth: float = 0.12,
                terminal_growth: float = 0.03,
                wacc: float = None,
                risk_free_rate: float = 0.065,
                equity_risk_premium: float = 0.055,
                use_mid_year_discount: bool = True,
                **kwargs) -> DCFResult:
        """
        Run full 3-stage DCF with proper discounting
        """
        try:
            stage1_years = max(1, int(stage1_years or 0))
            stage2_years = max(1, int(stage2_years or 0))
        except Exception:
            stage1_years = 3
            stage2_years = 4

        # Calculate WACC if not provided
        if wacc is None:
            wacc = self.calculate_wacc(risk_free_rate, equity_risk_premium)

        # Validate assumptions
        if stage1_growth < 0.02:
            _banner("warning", "Growth Check", f"Stage 1 growth ({stage1_growth:.1%}) seems low - typical is 8-20%")

        if terminal_growth > 0.07:
            _banner("warning", "Terminal Growth Check", f"Terminal growth ({terminal_growth:.1%}) exceeds Indian GDP growth - unrealistic")

        if terminal_growth >= wacc:
            _banner("error", "Terminal Growth Error", f"Terminal growth ({terminal_growth:.1%}) cannot exceed WACC ({wacc:.1%}); adjusting.")
            terminal_growth = wacc * 0.8
        
        # Get projections
        revenues, ebists, fcfs = self.project_fcf_3stage(
            stage1_years=stage1_years,
            stage2_years=stage2_years,
            stage1_growth=stage1_growth,
            terminal_growth=terminal_growth,
            **kwargs
        )

        if not fcfs:
            raise ValueError(
                "DCF projection produced no free cash flows. Check revenue, growth, and stage-year assumptions."
            )
        
        total_years = len(fcfs)
        
        # Discount FCFs
        pv_stage1 = 0
        pv_stage2 = 0
        
        for i, fcf in enumerate(fcfs):
            if use_mid_year_discount:
                # Mid-year convention (more accurate)
                discount_factor = 1 / (1 + wacc) ** (i + 0.5)
            else:
                discount_factor = 1 / (1 + wacc) ** (i + 1)
            
            if i < stage1_years:
                pv_stage1 += fcf * discount_factor
            else:
                pv_stage2 += fcf * discount_factor
        
        # Terminal Value using Gordon Growth model
        last_fcf = fcfs[-1]
        if wacc <= terminal_growth:
            # This shouldn't happen with reasonable assumptions
            terminal_value = last_fcf * 20  # Fallback: 20x multiple
        else:
            terminal_value = (last_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        
        # Discount terminal value
        if use_mid_year_discount:
            terminal_discount = 1 / (1 + wacc) ** (total_years - 0.5)
        else:
            terminal_discount = 1 / (1 + wacc) ** total_years
        
        pv_terminal = terminal_value * terminal_discount
        
        # Enterprise Value
        enterprise_value = pv_stage1 + pv_stage2 + pv_terminal
        
        # Get net debt and other adjustments
        total_debt = self.metrics.get("total_debt", 0)
        cash = self.metrics.get("cash", 0)
        net_debt = max(0, total_debt - cash)  # Ensure non-negative
        minorities = self.metrics.get("minority_interest", 0)
        
        # Equity Value
        equity_value = max(0, enterprise_value - net_debt - minorities)
        
        # Per Share Value
        shares = self.company.info.get("sharesOutstanding")
        if not shares or shares <= 0:
            # Estimate from market cap
            market_cap = self.metrics.get("market_cap")
            current_price = self.metrics.get("current_price", 100)
            if market_cap and market_cap > 0 and current_price > 0:
                shares = market_cap / current_price
            else:
                # Fallback: assume 1B shares for large Indian companies
                shares = 1e9
        
        per_share = equity_value / shares if shares > 0 else 0
        
        # Current price and premium
        current_price = self.metrics.get("current_price")
        if not current_price or current_price <= 0:
            current_price = per_share * 0.9
        
        implied_premium = ((per_share / current_price) - 1) * 100 if current_price > 0 else 0
        
        # Fairness rating
        if implied_premium > 20:
            fairness = "Significantly Undervalued"
        elif implied_premium > 10:
            fairness = "Moderately Undervalued"
        elif implied_premium > -10:
            fairness = "Fairly Valued"
        elif implied_premium > -20:
            fairness = "Moderately Overvalued"
        else:
            fairness = "Significantly Overvalued"
        
        # Run sensitivity analysis
        sensitivity = self._run_sensitivity_analysis(wacc, terminal_growth, per_share)
        
        # Terminal value warning
        terminal_pct = (pv_terminal / enterprise_value) * 100 if enterprise_value > 0 else 0
        
        return DCFResult(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            per_share=per_share,
            stage1_pv=pv_stage1,
            stage2_pv=pv_stage2,
            terminal_pv=pv_terminal,
            terminal_value=terminal_value,
            projection_years=list(range(1, total_years + 1)),
            revenue_projections=revenues,
            ebit_projections=ebists,  # FIXED: Now included
            fcf_projections=fcfs,
            wacc=wacc,
            terminal_growth=terminal_growth,
            stage1_growth=stage1_growth,
            stage2_growth_decline=(stage1_growth - terminal_growth) / stage2_years if stage2_years > 0 else 0,
            current_price=current_price,
            implied_premium=implied_premium,
            fairness_rating=fairness,
            sensitivity_matrix=sensitivity
        )