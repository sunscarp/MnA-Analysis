"""
AI-powered assumption generation using Groq LLM
"""
import os
import json
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import streamlit as st
from groq import Groq
from src.assumptions import ValuationAssumptions

# Indian market constants
INDIA_10Y_YIELD = 0.071  # 7.1% as of May 2026
INDIA_GDP_GROWTH = 0.065  # 6.5%
INDIA_INFLATION = 0.045   # 4.5%

# Sector-specific system prompts
SECTOR_SYSTEM_PROMPTS = {
    "airlines": """You are an expert M&A investment banker specializing in AVIATION / AIRLINES sector in India.
Key considerations for airlines:
- High operating leverage (fuel ~30-40% of costs)
- Cyclical demand tied to GDP and disposable income
- Intense price competition and low margins (5-10% EBITDA typical)
- High capex for fleet expansion
- Fuel price sensitivity (Brent crude, USD/INR)
- Seasonality and load factor dynamics
- IndiGo specific: Market leader with 60%+ market share, young fleet, strong balance sheet
- Terminal growth typically lower than GDP due to market maturity
- Control premiums typically 20-35% for airlines""",

    "technology": """You are an expert M&A investment banker specializing in TECHNOLOGY / IT SERVICES sector in India.
Key considerations for IT services:
- High margins (20-30% EBITDA typical)
- Lower capital intensity
- People-dependent business (high employee costs)
- Currency sensitivity (USD/INR)
- Growth tied to global tech spending and digital transformation
- Infosys specific: Tier-1 player with strong cash flows, high ROE
- Terminal growth 4-6% (above GDP due to global addressable market)
- Control premiums typically 25-40% for quality tech assets""",

    "financial": """You are an expert M&A investment banker specializing in FINANCIAL SERVICES / BANKING sector in India.
Key considerations for banks/NBFCs:
- NIM (Net Interest Margin) based valuation
- Asset quality (GNPA, NNPA) critical
- Regulatory capital requirements
- Price/Book value primary metric (1.5-3.0x typical)
- ROE driven valuation (12-18% typical)
- Terminal growth tied to credit growth + GDP
- Control premiums typically 15-30% for banks""",

    "energy": """You are an expert M&A investment banker specializing in ENERGY / OIL & GAS sector in India.
Key considerations for energy:
- Commodity price cyclicality (crude oil, gas)
- Government regulation and subsidy impacts
- High capex intensity
- Refining margins volatility
- Reliance specific: Diversified (O2C, Retail, Digital, Jio)
- Terminal growth 3-5% (mature industry)
- Control premiums typically 15-25%""",

    "consumer": """You are an expert M&A investment banker specializing in CONSUMER GOODS / FMCG sector in India.
Key considerations for consumer/FMCG:
- Brand strength and distribution network key assets
- Stable margins (15-20% EBITDA typical)
- Moderate growth (8-12%) tied to consumption
- Working capital intensity
- Premium valuation multiples (PE 30-40x for leaders)
- Terminal growth 5-7% (brands outlast GDP)
- Control premiums typically 25-40% for strong brands""",

    "industrial": """You are an expert M&A investment banker specializing in INDUSTRIALS / CAPITAL GOODS sector in India.
Key considerations for industrials:
- Order book visibility drives valuation
- Cyclical margins (10-18% EBITDA)
- High capex and working capital needs
- Infrastructure and government spending linked
- L&T specific: EPC giant with international presence
- Terminal growth 4-6%
- Control premiums typically 20-35%""",

    "general": """You are an expert M&A investment banker with 15+ years experience in Indian M&A.
Generate realistic, defensible valuation assumptions for a potential acquisition.
Consider sector-specific dynamics, Indian macro context, and historical deal precedents.
Be conservative/realistic unless specified otherwise."""
}


@dataclass
class AIAssumptionResult:
    """Container for AI-generated assumptions with validation"""
    assumptions: Dict[str, Any]
    rationales: Dict[str, str]
    confidence: str
    key_risks: list
    raw_response: str
    success: bool
    error: str = ""


class AIAssumptionGenerator:
    """Generate valuation assumptions using Groq LLM"""
    
    def __init__(self, api_key: str = None):
        """Initialize Groq client"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in environment or pass directly.")
        
        self.client = Groq(api_key=self.api_key)
    
    def _get_sector_prompt(self, sector: str) -> str:
        """Get sector-specific system prompt"""
        sector_lower = sector.lower()
        for key, prompt in SECTOR_SYSTEM_PROMPTS.items():
            if key in sector_lower:
                return prompt
        return SECTOR_SYSTEM_PROMPTS["general"]
    
    def _build_financial_summary(self, company) -> str:
        """Build structured financial summary for LLM context"""
        metrics = company.get_key_metrics()
        bs = company.get_balance_sheet_metrics() if hasattr(company, "get_balance_sheet_metrics") else {}
        
        summary = f"""
### Target: {metrics.get('name', company.ticker)} ({company.ticker})
- Sector: {metrics.get('sector', 'N/A')}
- Market Cap: ₹{metrics.get('market_cap', 0)/1e9:.1f}B
- Revenue (TTM): ₹{metrics.get('revenue', 0)/1e9:.1f}B
- EBITDA (TTM): ₹{metrics.get('ebitda', 0)/1e9:.1f}B
- EBITDA Margin: {metrics.get('ebitda_margin', 0)*100:.1f}%
- Net Income: ₹{metrics.get('net_income', 0)/1e9:.1f}B
- Net Debt: ₹{(metrics.get('total_debt', 0) - metrics.get('cash', 0))/1e9:.1f}B
- Current P/E: {metrics.get('pe_ratio', 0):.1f}x
- Current EV/EBITDA: {metrics.get('ev_ebitda', 0):.1f}x
- Current Price: ₹{metrics.get('current_price', 0):.2f}
- Beta: {metrics.get('beta', 1.0):.2f}

Balance Sheet:
- Total Assets: ₹{bs.get('total_assets', 0)/1e9:.1f}B
- Total Debt: ₹{bs.get('total_debt', 0)/1e9:.1f}B
- Cash: ₹{bs.get('cash', 0)/1e9:.1f}B
- Book Value: ₹{bs.get('book_value', 0)/1e9:.1f}B
"""
        return summary
    
    def _build_macro_context(self, sector: str) -> str:
        """Build macro-economic context"""
        return f"""
### Indian Macro Context (May 2026)
- 10-Year G-Sec Yield: {INDIA_10Y_YIELD:.1%}
- GDP Growth Forecast: {INDIA_GDP_GROWTH:.1%}
- Inflation (CPI): {INDIA_INFLATION:.1%}
- Policy Repo Rate: ~6.0-6.5% (implied)
- USD/INR: ~83-85 range

### Sector Context ({sector})
Refer to the expert system prompt above for sector-specific considerations.
"""
    
    def _build_prompt(self, company, acquirer, sector: str, user_mode: str = "base") -> str:
        """Build complete prompt for Groq"""
        target_summary = self._build_financial_summary(company)
        macro_context = self._build_macro_context(sector)
        
        # Add acquirer context if available
        acquirer_summary = ""
        if acquirer and hasattr(acquirer, 'success') and acquirer.success:
            a_metrics = acquirer.get_key_metrics()
            acquirer_summary = f"""
### Acquirer Context ({a_metrics.get('name', acquirer.ticker)})
- Revenue: ₹{a_metrics.get('revenue', 0)/1e9:.1f}B
- EBITDA Margin: {a_metrics.get('ebitda_margin', 0)*100:.1f}%
- Market Cap: ₹{a_metrics.get('market_cap', 0)/1e9:.1f}B
- Strategic Rationale: Potential vertical integration / market expansion / diversification
"""
        
        mode_instructions = {
            "conservative": "Use conservative assumptions (lower growth, higher WACC, lower multiples, lower synergies).",
            "base": "Use realistic, data-driven assumptions (mid-range estimates based on historicals and peers).",
            "optimistic": "Use optimistic but still plausible assumptions (higher growth, lower WACC, higher multiples, full synergies).",
            "strategic": "Assume strategic buyer with higher synergies and control premium (20-40% above base)."
        }
        
        prompt = f"""You are a senior M&A Investment Banker at a top-tier Indian firm (e.g., JPMorgan / Goldman Sachs India) with 18+ years of experience in cross-sector deals on the NSE/BSE.

Your job is to generate realistic, defensible, and conservative-to-balanced valuation assumptions for a potential acquisition of the target by the acquirer.

Current Date: May 2026

### Input Data
{target_summary}
{acquirer_summary}
{macro_context}

### User Mode
{user_mode.upper()} - {mode_instructions.get(user_mode, mode_instructions['base'])}

### CRITICAL FORMATTING RULES:
1. ALL numeric values MUST be decimals (e.g., 0.12 for 12%, NOT 12)
2. Do NOT use percentage signs - use decimals only
3. Do NOT use mathematical expressions - output only final calculated numbers
4. Growth rates, margins, premiums should be between 0 and 1
5. Multiples (EV/EBITDA, P/E) can be >1

### Output Schema (use EXACT field names):

Output ONLY valid JSON with this exact schema:

{{
    "company_summary": "2-3 sentence overview of target and strategic fit",
    "assumptions": {{
        "macro": {{
            "risk_free_rate": 0.071,
            "equity_risk_premium": 0.055,
            "tax_rate": 0.25
        }},
        "dcf": {{
            "stage1_years": 3,
            "stage2_years": 4,
            "stage1_growth": 0.12,
            "terminal_growth": 0.05,
            "stage1_ebitda_margin": 0.18,
            "terminal_margin": 0.15,
            "reinvestment_rate": 0.35
        }},
        "trading_comps": {{
            "ev_revenue": 3.2,
            "ev_ebitda": 14.5,
            "pe_ratio": 24.0
        }},
        "precedent": {{
            "control_premium": 0.25,
            "transaction_ev_ebitda": 16.0
        }},
        "wacc": {{
            "target_de_ratio": 0.30
        }}
    }},
    "rationales": {{
        "stage1_growth": "Based on historical CAGR and sector outlook...",
        "terminal_growth": "Long-term GDP growth assumption...",
        "ev_ebitda_multiple": "Peer group median multiple...",
        "control_premium": "Recent M&A transactions in sector..."
    }},
    "confidence": "High/Medium/Low",
    "key_risks": [
        "Risk 1 with mitigation",
        "Risk 2 with mitigation"
    ]
}}

Generate your response NOW (JSON only, no other text):
"""
        return prompt

    def _normalize_assumptions(self, assumptions: Dict) -> Dict:
        """Convert percentage-style AI outputs to decimal values used by the model."""
        normalized = dict(assumptions) if assumptions else {}
        
        # Recursively process all numeric values
        def normalize_value(value):
            if isinstance(value, dict):
                return {k: normalize_value(v) for k, v in value.items()}
            elif isinstance(value, (int, float)):
                # If value > 1 and looks like a percentage (between 1 and 100)
                if 1 < value <= 100:
                    return value / 100
                return value
            return value
        
        return normalize_value(normalized)
    
    def _validate_assumptions(self, assumptions: Dict) -> Tuple[bool, list]:
        """Validate generated assumptions against reasonable bounds"""
        warnings = []
        is_valid = True
        
        # Macro bounds
        macro = assumptions.get("macro", {})
        if macro.get("risk_free_rate", 0) < 0.03 or macro.get("risk_free_rate", 0) > 0.12:
            warnings.append(f"Risk-free rate {macro.get('risk_free_rate', 0):.1%} outside reasonable range (3-12%)")
        if macro.get("equity_risk_premium", 0) < 0.04 or macro.get("equity_risk_premium", 0) > 0.08:
            warnings.append(f"ERP {macro.get('equity_risk_premium', 0):.1%} outside reasonable range (4-8%)")
        
        # DCF bounds
        dcf = assumptions.get("dcf", {})
        if dcf.get("terminal_growth", 0) > 0.07:
            warnings.append(f"Terminal growth {dcf.get('terminal_growth', 0):.1%} exceeds Indian GDP growth (6.5%)")
            is_valid = False
        if dcf.get("terminal_growth", 0) < 0.02:
            warnings.append(f"Terminal growth {dcf.get('terminal_growth', 0):.1%} seems too low")
        if dcf.get("stage1_growth", 0) < 0.05 or dcf.get("stage1_growth", 0) > 0.35:
            warnings.append(f"Stage 1 growth {dcf.get('stage1_growth', 0):.1%} outside reasonable range (5-35%)")
        if dcf.get("stage1_ebitda_margin", 0) < 0.05 or dcf.get("stage1_ebitda_margin", 0) > 0.50:
            warnings.append(f"EBITDA margin {dcf.get('stage1_ebitda_margin', 0):.1%} outside reasonable range (5-50%)")
        
        # Comps bounds
        comps = assumptions.get("trading_comps", assumptions.get("comps", {}))
        if comps.get("ev_ebitda", 0) < 4 or comps.get("ev_ebitda", 0) > 30:
            warnings.append(f"EV/EBITDA multiple {comps.get('ev_ebitda', 0):.1f}x outside range (4-30x)")
        
        # Precedent bounds
        precedent = assumptions.get("precedent", {})
        if precedent.get("control_premium", 0) < 0.10 or precedent.get("control_premium", 0) > 0.60:
            warnings.append(f"Control premium {precedent.get('control_premium', 0):.1%} outside range (10-60%)")
        
        return is_valid, warnings
    
    def generate_assumptions(self, 
                            company, 
                            acquirer=None, 
                            sector: str = "general",
                            user_mode: str = "base",
                            temperature: float = 0.3,
                            max_retries: int = 2) -> AIAssumptionResult:
        """
        Generate valuation assumptions using Groq
        """
        sector_prompt = self._get_sector_prompt(sector)
        
        system_prompt = sector_prompt + """
        
Always output valid JSON only. Do not include any explanatory text outside the JSON.
Use the exact schema provided. CRITICAL: All growth rates, margins, and premiums must be DECIMALS (e.g., 0.12 for 12%).
"""
        
        user_prompt = self._build_prompt(company, acquirer, sector, user_mode)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )
                
                raw_response = response.choices[0].message.content

                # Try to parse JSON robustly
                data = None
                try:
                    data = json.loads(raw_response)
                except Exception:
                    # If response is already a dict-like object
                    if isinstance(raw_response, dict):
                        data = raw_response
                    else:
                        raise

                # Robust assumptions extraction: support multiple schema variants
                assumptions_data = {}
                if isinstance(data, dict):
                    # Preferred: top-level 'assumptions' key
                    if isinstance(data.get('assumptions'), dict):
                        assumptions_data = data.get('assumptions')

                    # Some models put sections at top-level (dcf, macro, trading_comps, etc.)
                    expected_sections = ['macro', 'wacc', 'dcf', 'trading_comps', 'comps', 'precedent', 'synergies']
                    for sec in expected_sections:
                        if sec in data and isinstance(data[sec], dict):
                            assumptions_data.setdefault(sec, {}).update(data[sec])

                    # Fallback: singular 'assumption' or nested JSON string
                    if not assumptions_data and isinstance(data.get('assumption'), dict):
                        assumptions_data = data.get('assumption')
                    if not assumptions_data and isinstance(data.get('assumptions'), str):
                        try:
                            parsed = json.loads(data.get('assumptions'))
                            if isinstance(parsed, dict):
                                assumptions_data = parsed
                        except Exception:
                            pass

                # Normalize numeric representations (percent -> decimals)
                if isinstance(assumptions_data, dict) and assumptions_data:
                    assumptions_data = self._normalize_assumptions(assumptions_data)

                # Rationales and risks with fallbacks
                rationales_data = {}
                if isinstance(data, dict):
                    rationales_data = data.get('rationales') or data.get('rationale') or data.get('explanations') or {}

                confidence_value = (data.get('confidence') or data.get('confidence_level') or data.get('confidenceLevel') or 'Medium') if isinstance(data, dict) else 'Medium'

                key_risks_value = []
                if isinstance(data, dict):
                    key_risks_value = data.get('key_risks') or data.get('keyRisks') or data.get('key_risks_list') or []
                    if not key_risks_value and isinstance(rationales_data, dict):
                        key_risks_value = rationales_data.get('risks') or rationales_data.get('key_risks') or []

                # If we couldn't find any assumptions, fail clearly so we can retry or debug
                if not assumptions_data:
                    raise ValueError('Missing required key: assumptions (no recognizable assumptions found in model output)')

                # Validate (non-fatal warnings printed)
                is_valid, warnings = self._validate_assumptions(assumptions_data)
                if warnings:
                    print(f"AI Assumptions warnings: {warnings}")

                return AIAssumptionResult(
                    assumptions=assumptions_data,
                    rationales=rationales_data or {},
                    confidence=confidence_value,
                    key_risks=key_risks_value or [],
                    raw_response=raw_response,
                    success=True,
                    error=""
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error: {e}"
                if attempt == max_retries - 1:
                    return AIAssumptionResult(
                        assumptions={},
                        rationales={},
                        confidence="Low",
                        key_risks=[],
                        raw_response=raw_response if 'raw_response' in locals() else "",
                        success=False,
                        error=error_msg
                    )
                    
            except Exception as e:
                error_msg = str(e)
                if attempt == max_retries - 1:
                    return AIAssumptionResult(
                        assumptions={},
                        rationales={},
                        confidence="Low",
                        key_risks=[],
                        raw_response="",
                        success=False,
                        error=error_msg
                    )
        
        return AIAssumptionResult(
            assumptions={},
            rationales={},
            confidence="Low",
            key_risks=[],
            raw_response="",
            success=False,
            error="Max retries exceeded"
        )


def apply_ai_assumptions_to_model(assumptions: ValuationAssumptions, 
                                  ai_result: AIAssumptionResult) -> ValuationAssumptions:
    """Apply AI-generated assumptions to the ValuationAssumptions object"""
    if not ai_result.success:
        return assumptions
    
    ai_assumptions = ai_result.assumptions or {}
    
    # Direct mapping of AI output fields to ValuationAssumptions fields
    updates = {}
    
    # Macro section
    macro = ai_assumptions.get("macro", {})
    if "risk_free_rate" in macro and macro["risk_free_rate"] is not None:
        updates["risk_free_rate"] = macro["risk_free_rate"]
    if "equity_risk_premium" in macro and macro["equity_risk_premium"] is not None:
        updates["equity_risk_premium"] = macro["equity_risk_premium"]
    if "tax_rate" in macro and macro["tax_rate"] is not None:
        updates["tax_rate"] = macro["tax_rate"]
    
    # DCF section
    dcf = ai_assumptions.get("dcf", {})
    
    if "stage1_growth" in dcf and dcf["stage1_growth"] is not None:
        updates["stage1_growth"] = dcf["stage1_growth"]
    if "terminal_growth" in dcf and dcf["terminal_growth"] is not None:
        updates["terminal_growth"] = dcf["terminal_growth"]
    if "stage1_years" in dcf and dcf["stage1_years"] is not None:
        updates["stage1_years"] = max(1, int(dcf["stage1_years"]))
    if "stage2_years" in dcf and dcf["stage2_years"] is not None:
        updates["stage2_years"] = max(1, int(dcf["stage2_years"]))
    if "stage1_ebitda_margin" in dcf and dcf["stage1_ebitda_margin"] is not None:
        updates["stage1_ebitda_margin"] = dcf["stage1_ebitda_margin"]
    if "terminal_margin" in dcf and dcf["terminal_margin"] is not None:
        updates["terminal_margin"] = dcf["terminal_margin"]
    if "reinvestment_rate" in dcf and dcf["reinvestment_rate"] is not None:
        updates["reinvestment_rate"] = dcf["reinvestment_rate"]
    
    # Trading Comps section
    comps = ai_assumptions.get("trading_comps", ai_assumptions.get("comps", {}))
    if "ev_ebitda" in comps and comps["ev_ebitda"] is not None:
        updates["ev_ebitda_multiple"] = comps["ev_ebitda"]
    if "pe_ratio" in comps and comps["pe_ratio"] is not None:
        updates["pe_multiple"] = comps["pe_ratio"]
    if "ev_revenue" in comps and comps["ev_revenue"] is not None:
        updates["ev_revenue_multiple"] = comps["ev_revenue"]
    
    # Precedent section
    precedent = ai_assumptions.get("precedent", {})
    if "control_premium" in precedent and precedent["control_premium"] is not None:
        updates["control_premium"] = precedent["control_premium"]
    if "transaction_ev_ebitda" in precedent and precedent["transaction_ev_ebitda"] is not None:
        updates["transaction_ev_ebitda"] = precedent["transaction_ev_ebitda"]
    
    # WACC section
    wacc_section = ai_assumptions.get("wacc", {})
    if "target_de_ratio" in wacc_section and wacc_section["target_de_ratio"] is not None:
        de_ratio = wacc_section["target_de_ratio"]
        if isinstance(de_ratio, (int, float)):
            # Convert D/E ratio to debt-to-capital if needed
            updates["debt_to_capital"] = de_ratio / (1 + de_ratio)
    
    # Apply all updates
    for key, value in updates.items():
        if hasattr(assumptions, key) and value is not None:
            setattr(assumptions, key, value)
    
    # Debug output
    print(f"AI Assumptions applied: {list(updates.keys())}")
    if "stage1_growth" in updates:
        print(f"  stage1_growth: {updates['stage1_growth']:.1%}")
    if "terminal_growth" in updates:
        print(f"  terminal_growth: {updates['terminal_growth']:.1%}")
    if "ev_ebitda_multiple" in updates:
        print(f"  ev_ebitda_multiple: {updates['ev_ebitda_multiple']:.1f}x")
    
    return assumptions