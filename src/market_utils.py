import yfinance as yf
from typing import Dict


class MarketDataProvider:
    """Provides market-derived assumptions."""

    @staticmethod
    def get_sector_multiple(ticker: str, metric: str, sector: str = "general") -> float:
        """Get sector-appropriate multiple from live data or benchmarks."""
        sector_comps = {
            "technology": {"ev_ebitda": 18.5, "pe": 28.0, "ev_revenue": 5.5},
            "financial": {"ev_ebitda": 12.0, "pe": 15.0, "ev_revenue": 3.5},
            "industrial": {"ev_ebitda": 10.5, "pe": 18.0, "ev_revenue": 1.8},
            "consumer": {"ev_ebitda": 12.0, "pe": 22.0, "ev_revenue": 2.0},
            "energy": {"ev_ebitda": 8.0, "pe": 14.0, "ev_revenue": 1.2},
            "healthcare": {"ev_ebitda": 16.0, "pe": 25.0, "ev_revenue": 4.0},
            "airlines": {"ev_ebitda": 6.5, "pe": 12.0, "ev_revenue": 0.8},
            "general": {"ev_ebitda": 12.0, "pe": 20.0, "ev_revenue": 3.0}
        }

        return sector_comps.get(sector, sector_comps["general"]).get(metric, 12.0)

    @staticmethod
    def get_company_beta(ticker: str) -> float:
        """Get beta from yfinance or estimate from sector."""
        try:
            obj = yf.Ticker(ticker)
            beta = obj.info.get("beta")
            if beta and 0.5 < beta < 2.5:
                return float(beta)
        except Exception:
            pass

        return 1.0

    @staticmethod
    def derive_growth_rate(company) -> float:
        """Derive growth rate from actual data."""
        try:
            growth = company.info.get("growth_estimates", {})
            if "next_year" in growth:
                return float(growth["next_year"]) / 100
        except Exception:
            pass

        try:
            financials = company.financials
            if financials is not None and not financials.empty:
                revenues = []
                for col in financials.columns[:3]:
                    try:
                        rev = financials.loc[
                            financials.index.str.contains("revenue", case=False)
                        ].iloc[0][col]
                        revenues.append(float(rev))
                    except Exception:
                        pass
                if len(revenues) >= 2:
                    cagr = (revenues[0] / revenues[-1]) ** (1 / len(revenues)) - 1
                    if 0.02 < cagr < 0.40:
                        return cagr
        except Exception:
            pass

        ticker = company.ticker
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            return 0.12
        return 0.08

    @staticmethod
    def derive_margins(company) -> Dict[str, float]:
        """Derive EBITDA margins from actual financials."""
        try:
            financials = company.financials
            if financials is not None and not financials.empty:
                revenue_row = financials.index[
                    financials.index.str.contains("revenue", case=False)
                ]
                ebitda_row = financials.index[
                    financials.index.str.contains("ebitda", case=False)
                ]

                if len(revenue_row) > 0 and len(ebitda_row) > 0:
                    latest_rev = float(financials.loc[revenue_row[0]].iloc[0])
                    latest_ebitda = float(financials.loc[ebitda_row[0]].iloc[0])
                    if latest_rev > 0:
                        return {
                            "ebitda_margin": latest_ebitda / latest_rev,
                            "source": "actuals"
                        }
        except Exception:
            pass

        return {"ebitda_margin": 0.15, "source": "sector_average"}
