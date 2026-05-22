from typing import List

from src.dcf_model import DCFModel, DCFResult
from src.comps_model import CompsModel, CompsResult
from src.precedent_model import PrecedentModel, PrecedentResult
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

class ValuationModel:
    """Orchestrator for all valuation methods"""
    
    def __init__(self, company, sector: str = "general"):
        self.company = company
        self.sector = sector
        self.dcf_model = DCFModel(company, sector)
        self.comps_model = CompsModel(company, sector)
        self.precedent_model = PrecedentModel(company, sector)
    
    def run_dcf(self, **kwargs) -> DCFResult:
        """Run DCF with given assumptions"""
        return self.dcf_model.run_dcf(**kwargs)
    
    def calculate_trading_comps(self, **kwargs) -> CompsResult:
        """Calculate trading comps valuation"""
        return self.comps_model.calculate_trading_comps(**kwargs)
    
    def calculate_precedent_transactions(self, **kwargs) -> PrecedentResult:
        """Calculate precedent transactions valuation"""
        return self.precedent_model.calculate_precedent_value(**kwargs)

    def validate_assumptions(self, dcf_result: DCFResult) -> List[str]:
        """Validate assumptions and return warnings"""
        warnings = []

        current_price = dcf_result.current_price or 0
        if current_price < 100 or current_price > 50000:
            warnings.append(
                f"WARNING: Current price (INR {current_price:,.0f}) seems unusual - verify data source"
            )

        if hasattr(self.dcf_model, "risk_free_rate"):
            if self.dcf_model.risk_free_rate < 0.03:
                warnings.append(
                    f"WARNING: Risk-free rate ({self.dcf_model.risk_free_rate:.1%}) is too low - should be ~6.5% for India"
                )

        if dcf_result.terminal_growth > 0.07:
            warnings.append(
                f"WARNING: Terminal growth ({dcf_result.terminal_growth:.1%}) exceeds Indian GDP growth"
            )

        if dcf_result.terminal_growth >= dcf_result.wacc:
            warnings.append(
                f"ERROR: Terminal growth ({dcf_result.terminal_growth:.1%}) >= WACC ({dcf_result.wacc:.1%}) - invalid"
            )

        terminal_pct = (
            (dcf_result.terminal_pv / dcf_result.enterprise_value) * 100
            if dcf_result.enterprise_value > 0
            else 0
        )
        if terminal_pct > 70:
            warnings.append(
                f"WARNING: Terminal value is {terminal_pct:.0f}% of DCF - sensitive to terminal growth assumption"
            )

        return warnings
    
    def create_football_field(self,
                             dcf_result: DCFResult,
                             comps_result: CompsResult,
                             precedent_result: PrecedentResult,
                             show_target_only: bool = True) -> go.Figure:
        """Create professional football field chart"""
        
        dcf_value = dcf_result.per_share
        comps_value = comps_result.per_share_weighted
        precedent_value = precedent_result.per_share_with_premium
        current_price = dcf_result.current_price
        
        dcf_low = dcf_value * 0.85
        dcf_high = dcf_value * 1.15
        
        comps_low = comps_result.per_share_range[0] if comps_result.per_share_range[0] > 0 else comps_value * 0.8
        comps_high = comps_result.per_share_range[1] if comps_result.per_share_range[1] > 0 else comps_value * 1.2
        
        prec_low = precedent_result.per_share_range[0] if precedent_result.per_share_range[0] > 0 else precedent_value * 0.8
        prec_high = precedent_result.per_share_range[1] if precedent_result.per_share_range[1] > 0 else precedent_value * 1.2
        
        fig = go.Figure()
        
        methods = ["DCF", "Trading Comps", "Precedent Transactions"]
        lows = [dcf_low, comps_low, prec_low]
        highs = [dcf_high, comps_high, prec_high]
        colors = ["#00B4FF", "#0EA5E9", "#14B8A6"]
        
        for i, method in enumerate(methods):
            range_width = max(highs[i] - lows[i], 0)
            midpoint = lows[i] + (range_width / 2 if range_width > 0 else 0)
            fig.add_trace(go.Bar(
                y=[method],
                x=[range_width],
                base=[lows[i]],
                orientation='h',
                name=method,
                marker=dict(color=colors[i], opacity=0.7),
                text=[f"₹{lows[i]:,.0f} - ₹{highs[i]:,.0f}"],
                textposition='inside',
                insidetextanchor='middle',
                hovertemplate=(
                    f"<b>{method}</b><br>Low: ₹%{{{{base:,.0f}}}}<br>High: ₹%{{{{customdata:,.0f}}}}<extra></extra>"
                ),
                customdata=[highs[i]],
            ))

            fig.add_trace(go.Scatter(
                x=[lows[i], highs[i]],
                y=[method, method],
                mode='lines',
                line=dict(color=colors[i], width=6),
                showlegend=False,
                hoverinfo='skip',
            ))

            fig.add_trace(go.Scatter(
                x=[lows[i], highs[i]],
                y=[method, method],
                mode='markers',
                marker=dict(color=colors[i], size=8, symbol='line-ns-open'),
                showlegend=False,
                hoverinfo='skip',
            ))
        
        # Add reference lines
        if current_price and current_price > 0:
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="#00B4FF",
                line_width=2,
                annotation_text=f"Current: ₹{current_price:,.0f}",
                annotation_position="top right",
                annotation_font_color="#F1F5F9",
            )
        
        valid_values = [v for v in [dcf_value, comps_value, precedent_value] if v > 0]
        if valid_values:
            weighted_avg = np.mean(valid_values)
            fig.add_vline(
                x=weighted_avg,
                line_dash="dot",
                line_color="#94A3B8",
                line_width=2,
                annotation_text=f"Avg: ₹{weighted_avg:,.0f}",
                annotation_position="bottom right",
                annotation_font_color="#F1F5F9",
            )
        
        fig.update_layout(
            title="Valuation Football Field",
            xaxis_title="Value per Share (₹)",
            yaxis_title="Valuation Method",
            template="plotly_dark",
            height=350,
            barmode='overlay',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94A3B8")),
            paper_bgcolor="#0F1117",
            plot_bgcolor="#0A0C10",
            font=dict(color="#F1F5F9"),
            margin=dict(t=50, l=20, r=20, b=30),
            xaxis=dict(
                gridcolor="#1E2535",
                zerolinecolor="#1E2535",
                tickfont=dict(color="#94A3B8"),
                title_font=dict(color="#94A3B8"),
            ),
            yaxis=dict(
                gridcolor="#1E2535",
                categoryorder='array',
                categoryarray=methods,
                tickfont=dict(color="#94A3B8"),
                title_font=dict(color="#94A3B8"),
            ),
        )
        
        return fig
    
    def create_sensitivity_heatmap(self, dcf_result: DCFResult) -> go.Figure:
        """Create sensitivity analysis heatmap"""
        
        wacc_vars = dcf_result.sensitivity_matrix.get("wacc_variations", [])
        growth_vars = dcf_result.sensitivity_matrix.get("growth_variations", [])
        value_factors = dcf_result.sensitivity_matrix.get("values", {})
        
        if not wacc_vars:
            wacc_vars = [dcf_result.wacc - 0.01, dcf_result.wacc, dcf_result.wacc + 0.01]
        if not growth_vars:
            growth_vars = [dcf_result.terminal_growth - 0.005, dcf_result.terminal_growth, dcf_result.terminal_growth + 0.005]
        
        # Build matrix
        values = []
        for w in wacc_vars:
            row = []
            for g in growth_vars:
                if w in value_factors and g in value_factors.get(w, {}):
                    val = value_factors[w][g]
                else:
                    # Approximate
                    if w > 0:
                        val = dcf_result.per_share * (dcf_result.wacc / w) * ((1 + g) / (1 + dcf_result.terminal_growth))
                    else:
                        val = dcf_result.per_share
                row.append(val)
            values.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=[f"{g:.1%}" for g in growth_vars],
            y=[f"{w:.1%}" for w in wacc_vars],
            colorscale=[
                [0.0, '#450A0A'],
                [0.5, '#1E2535'],
                [1.0, '#064E3B'],
            ],
            text=np.round(values, 0),
            texttemplate='₹%{text}',
            textfont={"size": 11},
            colorbar_title="Per Share Value (₹)",
            hovertemplate="WACC: %{y}<br>Term Growth: %{x}<br>Value: ₹%{z:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="DCF Sensitivity: WACC vs Terminal Growth",
            xaxis_title="Terminal Growth Rate",
            yaxis_title="WACC",
            template="plotly_dark",
            height=450,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#0A0C10",
            font=dict(color="#F1F5F9"),
            xaxis=dict(gridcolor='#1E2535', tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8")),
            yaxis=dict(gridcolor='#1E2535', tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8"))
        )
        
        return fig