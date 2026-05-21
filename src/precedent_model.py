from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass
class PrecedentResult:
    """Precedent Transactions valuation results"""
    # Transaction multiples used
    median_ev_revenue: float
    median_ev_ebitda: float
    control_premium: float
    
    # Implied values
    implied_ev_range: Tuple[float, float]
    implied_equity_range: Tuple[float, float]
    per_share_range: Tuple[float, float]
    per_share_with_premium: float
    
    # Deal statistics
    selected_deals: List[Dict]
    premium_applied: float

class PrecedentModel:
    """Precedent M&A Transaction Analysis with control premium"""
    
    # Sector-specific transaction multiples (based on historical M&A)
    SECTOR_TRANSACTION_MULTIPLES = {
        "technology": {"ev_ebitda": 20.0, "ev_revenue": 6.0, "avg_premium": 0.35},
        "financial": {"ev_ebitda": 14.0, "ev_revenue": 4.0, "avg_premium": 0.25},
        "industrial": {"ev_ebitda": 12.0, "ev_revenue": 2.2, "avg_premium": 0.30},
        "consumer": {"ev_ebitda": 14.0, "ev_revenue": 2.5, "avg_premium": 0.32},
        "energy": {"ev_ebitda": 9.0, "ev_revenue": 1.5, "avg_premium": 0.28},
        "healthcare": {"ev_ebitda": 18.0, "ev_revenue": 4.5, "avg_premium": 0.40},
        "general": {"ev_ebitda": 14.0, "ev_revenue": 3.0, "avg_premium": 0.30}
    }
    
    def __init__(self, company, sector: str = "general"):
        self.company = company
        self.sector = sector
        self.metrics = company.get_key_metrics()
        self.sector_data = self.SECTOR_TRANSACTION_MULTIPLES.get(sector, self.SECTOR_TRANSACTION_MULTIPLES["general"])
    
    def calculate_precedent_value(self,
                                 custom_ev_ebitda: float = None,
                                 custom_ev_revenue: float = None,
                                 custom_control_premium: float = None) -> PrecedentResult:
        """
        Calculate valuation based on precedent M&A transactions
        """
        # Use custom or sector defaults
        ev_ebitda = custom_ev_ebitda if custom_ev_ebitda else self.sector_data["ev_ebitda"]
        ev_revenue = custom_ev_revenue if custom_ev_revenue else self.sector_data["ev_revenue"]
        control_premium = custom_control_premium if custom_control_premium else self.sector_data["avg_premium"]
        
        # Get financials
        revenue = self.metrics.get("revenue")
        ebitda = self.metrics.get("ebitda")
        
        if not revenue or revenue <= 0:
            revenue = ebitda * 6 if ebitda else 100e9
        
        if not ebitda or ebitda <= 0:
            ebitda = revenue * 0.15
        
        # Calculate EV with control premium
        ev_from_revenue = revenue * ev_revenue * (1 + control_premium)
        ev_from_ebitda = ebitda * ev_ebitda * (1 + control_premium)
        
        ev_values = [ev_from_revenue, ev_from_ebitda]
        implied_ev_min = min(ev_values)
        implied_ev_max = max(ev_values)
        
        # Calculate equity value
        net_debt = self.metrics.get("total_debt", 0) - self.metrics.get("cash", 0)
        equity_from_revenue = ev_from_revenue - net_debt
        equity_from_ebitda = ev_from_ebitda - net_debt
        
        equity_values = [equity_from_revenue, equity_from_ebitda]
        implied_equity_min = min(equity_values)
        implied_equity_max = max(equity_values)
        
        # Per share values
        shares = self.company.info.get("sharesOutstanding")
        if not shares or shares <= 0:
            shares = 1e9
        
        per_share_values = [eq / shares for eq in equity_values]
        per_share_min = min(per_share_values)
        per_share_max = max(per_share_values)
        
        # Average with premium
        avg_per_share = np.mean(per_share_values)
        
        return PrecedentResult(
            median_ev_revenue=ev_revenue,
            median_ev_ebitda=ev_ebitda,
            control_premium=control_premium,
            implied_ev_range=(implied_ev_min, implied_ev_max),
            implied_equity_range=(implied_equity_min, implied_equity_max),
            per_share_range=(per_share_min, per_share_max),
            per_share_with_premium=avg_per_share,
            selected_deals=[],
            premium_applied=control_premium
        )