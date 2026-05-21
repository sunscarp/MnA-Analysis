import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class CompsResult:
    """Trading Comps valuation results"""
    ev_revenue: float
    ev_ebitda: float
    pe_ratio: float
    pb_ratio: float
    implied_ev_range: Tuple[float, float]
    implied_equity_range: Tuple[float, float]
    per_share_range: Tuple[float, float]
    weighted_value: float
    per_share_weighted: float
    selected_comps: List[Dict]
    sector_median_ev_ebitda: float
    sector_median_pe: float


class CompsModel:
    """Comparable Company Analysis with sector benchmarks"""
    
    SECTOR_MULTIPLES = {
        "technology": {"ev_ebitda": 18.5, "pe": 28.0, "ev_revenue": 5.5, "pb": 6.0},
        "financial": {"ev_ebitda": 12.0, "pe": 15.0, "ev_revenue": 3.5, "pb": 1.8},
        "industrial": {"ev_ebitda": 10.5, "pe": 18.0, "ev_revenue": 1.8, "pb": 2.5},
        "consumer": {"ev_ebitda": 12.0, "pe": 22.0, "ev_revenue": 2.0, "pb": 3.5},
        "energy": {"ev_ebitda": 8.0, "pe": 14.0, "ev_revenue": 1.2, "pb": 1.5},
        "healthcare": {"ev_ebitda": 16.0, "pe": 25.0, "ev_revenue": 4.0, "pb": 4.5},
        "general": {"ev_ebitda": 12.0, "pe": 20.0, "ev_revenue": 2.5, "pb": 2.5}
    }
    
    def __init__(self, company, sector: str = "general"):
        self.company = company
        self.sector = sector
        self.metrics = company.get_key_metrics()
        self.sector_data = self.SECTOR_MULTIPLES.get(sector, self.SECTOR_MULTIPLES["general"])
    
    def calculate_trading_comps(self,
                               custom_ev_ebitda: float = None,
                               custom_pe: float = None,
                               custom_ev_revenue: float = None,
                               custom_pb: float = None,
                               confidence_weight: float = 0.6) -> CompsResult:
        """Calculate valuation using comparable company multiples"""
        
        ev_ebitda = custom_ev_ebitda if custom_ev_ebitda else self.sector_data["ev_ebitda"]
        pe_ratio = custom_pe if custom_pe else self.sector_data["pe"]
        ev_revenue = custom_ev_revenue if custom_ev_revenue else self.sector_data["ev_revenue"]
        pb_ratio = custom_pb if custom_pb else self.sector_data["pb"]
        
        revenue = self.metrics.get("revenue")
        ebitda = self.metrics.get("ebitda")
        net_income = self.metrics.get("net_income")
        book_value = self.metrics.get("book_value")
        
        if not revenue or revenue <= 0:
            revenue = ebitda * 5 if ebitda and ebitda > 0 else 100e9
        
        if not ebitda or ebitda <= 0:
            ebitda = revenue * (ev_ebitda / 15) if ev_ebitda > 0 else revenue * 0.15
        
        if not net_income or net_income <= 0:
            net_income = revenue * 0.10
        
        # Calculate implied EV
        ev_from_revenue = revenue * ev_revenue if ev_revenue else 0
        ev_from_ebitda = ebitda * ev_ebitda if ev_ebitda else 0
        
        ev_values = [v for v in [ev_from_revenue, ev_from_ebitda] if v > 0]
        implied_ev_min = min(ev_values) if ev_values else 0
        implied_ev_max = max(ev_values) if ev_values else 0
        
        # Calculate equity values
        net_debt = max(0, self.metrics.get("total_debt", 0) - self.metrics.get("cash", 0))
        
        equity_from_revenue = max(0, ev_from_revenue - net_debt) if ev_from_revenue > 0 else 0
        equity_from_ebitda = max(0, ev_from_ebitda - net_debt) if ev_from_ebitda > 0 else 0
        equity_from_pe = net_income * pe_ratio if pe_ratio else 0
        equity_from_pb = book_value * pb_ratio if book_value and pb_ratio else 0
        
        equity_values = [v for v in [equity_from_revenue, equity_from_ebitda, equity_from_pe, equity_from_pb] if v > 0]
        implied_equity_min = min(equity_values) if equity_values else 0
        implied_equity_max = max(equity_values) if equity_values else 0
        
        # Per share values
        shares = self.company.info.get("sharesOutstanding")
        if not shares or shares <= 0:
            current_price = self.metrics.get("current_price", 100)
            market_cap = self.metrics.get("market_cap")
            if market_cap and market_cap > 0 and current_price > 0:
                shares = market_cap / current_price
            else:
                shares = 1e9
        
        per_share_values = [eq / shares for eq in equity_values] if shares > 0 else []
        per_share_min = min(per_share_values) if per_share_values else 0
        per_share_max = max(per_share_values) if per_share_values else 0
        
        # Weighted average
        if equity_from_ebitda > 0:
            weighted_value = (
                equity_from_ebitda * confidence_weight +
                equity_from_pe * (1 - confidence_weight) * 0.7 +
                equity_from_revenue * (1 - confidence_weight) * 0.3
            )
        else:
            weighted_value = np.mean(equity_values) if equity_values else 0
        
        per_share_weighted = weighted_value / shares if shares > 0 else 0
        
        return CompsResult(
            ev_revenue=ev_revenue or 0,
            ev_ebitda=ev_ebitda or 0,
            pe_ratio=pe_ratio or 0,
            pb_ratio=pb_ratio or 0,
            implied_ev_range=(implied_ev_min, implied_ev_max),
            implied_equity_range=(implied_equity_min, implied_equity_max),
            per_share_range=(per_share_min, per_share_max),
            weighted_value=weighted_value,
            per_share_weighted=per_share_weighted,
            selected_comps=[],
            sector_median_ev_ebitda=self.sector_data["ev_ebitda"],
            sector_median_pe=self.sector_data["pe"]
        )