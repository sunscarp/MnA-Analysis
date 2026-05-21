import yfinance as yf
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, Optional
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

# ── Sheet name aliases ─────────────────────────────────────────────────────────
SHEET_ALIASES = {
    "income":   ["Income Statement", "Income_Statement", "P&L", "Financials", "IS"],
    "balance":  ["Balance Sheet",    "Balance_Sheet",    "BS", "BalanceSheet"],
    "cashflow": ["Cash Flow",        "Cash_Flow",        "CashFlow", "CF", "Cash Flows"],
    "info":     ["Info", "Overview", "Company Overview", "Summary"],
}

# Partial lowercase aliases for financial row matching
ROW_ALIASES = {
    "revenue":      ["total revenue", "net revenue", "net sales", "total sales"],
    "ebitda":       ["ebitda"],
    "ebit":         ["ebit", "operating income", "operating profit"],
    "net_income":   ["net income", "net profit", "profit after tax"],
    "da":           ["d&a", "depreciation"],
    "interest":     ["interest expense", "finance cost"],
    "gross_profit": ["gross profit"],
}

INFO_FIELD_MAP = {
    "company name":            "longName",
    "ticker symbol":           "symbol",
    "exchange":                "exchange",
    "sector":                  "sector",
    "currency":                "currency",
    "shares outstanding (mm)": "sharesOutstanding_mm",
    "current share price ($)": "currentPrice",
    "market cap ($mm)":        "marketCap_mm",
    "net debt ($mm)":          "netDebt_mm",
    "enterprise value ($mm)":  "enterpriseValue_mm",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_sheet(xls: pd.ExcelFile, key: str) -> Optional[str]:
    available_lower = {s.strip().lower(): s for s in xls.sheet_names}
    for alias in SHEET_ALIASES[key]:
        match = available_lower.get(alias.lower())
        if match:
            return match
    return None


def _detect_unit_multiplier(df: pd.DataFrame) -> float:
    """
    Detect the unit of financial values from row index labels.
    Looks for patterns like ($mm), ($bn), (₹ Cr), (₹ Lakh) in row labels.
    Returns a multiplier to convert to raw currency units.
    """
    idx_str = " ".join(str(i).lower() for i in df.index)
    if "$mm" in idx_str or "($mm)" in idx_str or "(usd mm)" in idx_str:
        return 1_000_000          # mm → raw
    if "$bn" in idx_str or "($bn)" in idx_str:
        return 1_000_000_000      # bn → raw
    if "cr" in idx_str or "crore" in idx_str:
        return 10_000_000         # Cr → raw INR
    if "lakh" in idx_str:
        return 100_000
    # No unit detected — assume values are already raw
    return 1.0


def _row_value(df: pd.DataFrame, key: str, multiplier: float = 1.0) -> Optional[float]:
    """
    Find a row by canonical key. Skips all-NaN rows.
    Prefers most-recent actual column (no 'E' suffix).
    Applies multiplier to convert stored units → raw.
    """
    aliases = ROW_ALIASES.get(key, [key.lower()])
    for idx in df.index:
        idx_clean = str(idx).strip().lower()
        if not any(alias in idx_clean for alias in aliases):
            continue
        try:
            vals = df.loc[idx].dropna()
            if vals.empty:
                continue
            actual_cols = [c for c in vals.index if "E" not in str(c)]
            chosen = float(vals[actual_cols[-1]] if actual_cols else vals.iloc[-1])
            return chosen * multiplier
        except Exception:
            continue
    return None


def _smart_read_sheet(xls: pd.ExcelFile, sheet_name: str, hint_words: list) -> pd.DataFrame:
    """Try header rows 1, 0, 2; return first with recognisable financial index labels."""
    for header_row in [1, 0, 2]:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row, index_col=0)
            idx_lower = [str(i).strip().lower() for i in df.index]
            if any(w in lbl for lbl in idx_lower for w in hint_words):
                return df
        except Exception:
            continue
    return pd.read_excel(xls, sheet_name=sheet_name, index_col=0)


def _parse_info_sheet(xls: pd.ExcelFile, sheet_name: str, existing_info: dict) -> dict:
    """Parse Company Overview / Info sheet. Supports index-label and Field/Value column formats."""
    info = dict(existing_info)

    # Format A: index = field labels, first data column = values  (our Excel)
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
        if df.shape[1] >= 1:
            for raw_label, raw_val in df.iloc[:, 0].items():
                label = str(raw_label).strip().lower()
                for field_key, canonical in INFO_FIELD_MAP.items():
                    if field_key in label:
                        info[canonical] = raw_val
                        break
    except Exception:
        pass

    # Format B: columns 'Field' + 'Value'
    try:
        df2 = pd.read_excel(xls, sheet_name=sheet_name)
        col_lower = [str(c).strip().lower() for c in df2.columns]
        if "field" in col_lower and "value" in col_lower:
            fi, vi = col_lower.index("field"), col_lower.index("value")
            for _, row in df2.iterrows():
                label = str(row.iloc[fi]).strip().lower()
                for field_key, canonical in INFO_FIELD_MAP.items():
                    if field_key in label:
                        info[canonical] = row.iloc[vi]
                        break
    except Exception:
        pass

    # Convert *_mm → raw (× 1,000,000)
    for mm_key, raw_key in [
        ("marketCap_mm",       "marketCap"),
        ("enterpriseValue_mm", "enterpriseValue"),
        ("netDebt_mm",         "netDebt"),
    ]:
        if mm_key in info:
            try:
                info[raw_key] = float(info[mm_key]) * 1_000_000
            except Exception:
                pass

    # Detect currency from share price or explicit field
    if "currency" not in info:
        # Infer from symbol — basic heuristic
        sym = str(info.get("symbol", ""))
        if sym.upper().endswith(".NS") or sym.upper().endswith(".BO"):
            info["currency"] = "INR"
        elif info.get("exchange", "") in ["NSE", "BSE"]:
            info["currency"] = "INR"
        else:
            info["currency"] = "USD"   # safe default for our sample Excel (USD-denominated)

    return info


# ── Company dataclass ──────────────────────────────────────────────────────────

@dataclass
class Company:
    ticker: str
    info: Dict = field(default_factory=dict)
    financials: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    success: bool = False
    error: str = ""
    source: str = "yfinance"
    # Multiplier: how many raw units per stored unit in financials
    _fin_multiplier: float = 1.0

    def __post_init__(self):
        if not self.success and self.ticker:
            self.fetch_all_data()

    def fetch_all_data(self):
        try:
            obj = yf.Ticker(self.ticker)
            self.info = obj.info or {}
            for attr, fetcher in [
                ("financials",    lambda: obj.income_stmt),
                ("balance_sheet", lambda: obj.balance_sheet),
                ("cash_flow",     lambda: obj.cash_flow),
            ]:
                try:
                    setattr(self, attr, fetcher())
                except Exception:
                    setattr(self, attr, None)
            # yfinance always returns raw numbers
            self._fin_multiplier = 1.0
            has_data = bool(self.info) or (
                self.financials is not None and not self.financials.empty
            )
            self.success = has_data
            self.source  = "yfinance"
            if not has_data:
                self.error = "No data returned. Try a different ticker or upload an Excel file."
        except Exception as e:
            self.error   = str(e)
            self.success = False

    def load_from_excel(self, excel_file) -> bool:
        try:
            xls       = pd.ExcelFile(excel_file)
            available = xls.sheet_names

            # ── Income Statement ──────────────────────────────────────────────
            income_sheet = _find_sheet(xls, "income")
            if income_sheet is None:
                income_sheet = available[0]
                _banner(
                    "warning",
                    "Income Sheet Not Found",
                    f"No 'Income Statement' sheet found. Using {escape(repr(income_sheet))}. Available: {escape(', '.join(available))}",
                )
            income = _smart_read_sheet(
                xls, income_sheet,
                ["revenue", "ebitda", "net income", "gross profit", "income", "sales"]
            )
            if income is None or income.empty:
                _banner("error", "Parse Error", f"Could not parse Income Statement from: {escape(repr(income_sheet))}")
                return False
            self.financials = income
            # Detect unit multiplier from row labels (e.g. "$mm" → 1_000_000)
            self._fin_multiplier = _detect_unit_multiplier(income)

            # ── Balance Sheet ─────────────────────────────────────────────────
            balance_name = _find_sheet(xls, "balance")
            if balance_name:
                try:
                    self.balance_sheet = _smart_read_sheet(
                        xls, balance_name, ["asset", "liabilit", "equity", "cash"]
                    )
                except Exception:
                    pass

            # ── Cash Flow ─────────────────────────────────────────────────────
            cf_name = _find_sheet(xls, "cashflow")
            if cf_name:
                try:
                    self.cash_flow = _smart_read_sheet(
                        xls, cf_name, ["operating", "investing", "financing", "capex"]
                    )
                except Exception:
                    pass

            # ── Info / Overview ───────────────────────────────────────────────
            info_sheet = _find_sheet(xls, "info")
            if info_sheet:
                self.info = _parse_info_sheet(xls, info_sheet, self.info)

            if not self.info.get("longName") and self.ticker:
                self.info["longName"] = self.ticker

            self.success = True
            self.source  = "excel"

            loaded  = ["Income Statement"]
            skipped = []
            if balance_name: loaded.append("Balance Sheet")
            else:            skipped.append("Balance Sheet")
            if cf_name:      loaded.append("Cash Flow")
            else:            skipped.append("Cash Flow")

            if skipped:
                _banner("info", "Loaded", f"Loaded: {escape(', '.join(loaded))}. Not found (optional): {escape(', '.join(skipped))}.")
            else:
                _banner("success", "Loaded from Excel", escape(', '.join(loaded)))
            return True

        except Exception as e:
            _banner("error", "Excel Load Failed", escape(str(e)))
            self.error = str(e)
            return False

    # ── Metric helpers ─────────────────────────────────────────────────────────

    def _fin_val(self, key: str) -> Optional[float]:
        """Get a financial row value, applying the correct unit multiplier for Excel files."""
        if self.financials is None or self.financials.empty:
            return None

        if self.source == "yfinance":
            aliases = ROW_ALIASES.get(key, [key.lower()])
            for idx in self.financials.index:
                idx_clean = str(idx).strip().lower()
                if any(a in idx_clean for a in aliases):
                    try:
                        vals = self.financials.loc[idx].dropna()
                        return float(vals.iloc[0]) if not vals.empty else None
                    except Exception:
                        pass
            return None
        else:
            # Excel: _row_value returns the stored number × multiplier → raw
            return _row_value(self.financials, key, self._fin_multiplier)

    def _info_num(self, *keys) -> Optional[float]:
        for k in keys:
            v = self.info.get(k)
            if v is not None:
                try:    return float(v)
                except: pass
        return None

    def get_balance_sheet_metrics(self) -> dict:
        """Get balance sheet data for PPA and pro forma analysis"""
        if not self.success:
            return {}

        # Get from yfinance info
        total_assets = self._info_num("totalAssets")
        total_liabilities = self._info_num("totalLiabilities")
        total_debt = self._info_num("totalDebt")
        cash = self._info_num("totalCash", "cash")

        # Try to get from balance sheet DataFrame if available
        if self.balance_sheet is not None and not self.balance_sheet.empty:
            try:
                # Look for common line items in balance sheet
                for idx in self.balance_sheet.index:
                    idx_str = str(idx).lower()
                    if "total assets" in idx_str and not total_assets:
                        total_assets = float(self.balance_sheet.loc[idx].iloc[0])
                    elif "total liabilities" in idx_str and not total_liabilities:
                        total_liabilities = float(self.balance_sheet.loc[idx].iloc[0])
                    elif "total debt" in idx_str and not total_debt:
                        total_debt = float(self.balance_sheet.loc[idx].iloc[0])
                    elif "cash" in idx_str and "cash and cash equivalents" in idx_str and not cash:
                        cash = float(self.balance_sheet.loc[idx].iloc[0])
            except Exception:
                pass

        # Book value
        book_value = None
        if total_assets and total_liabilities:
            book_value = total_assets - total_liabilities
        elif self._info_num("bookValue"):
            book_value = self._info_num("bookValue")

        # Tangible book value (exclude goodwill and intangibles)
        goodwill = self._info_num("goodwill")
        intangible_assets = self._info_num("intangibleAssets")

        tangible_book_value = book_value
        if tangible_book_value:
            if goodwill:
                tangible_book_value -= goodwill
            if intangible_assets:
                tangible_book_value -= intangible_assets

        # Working capital components
        current_assets = self._info_num("currentAssets")
        current_liabilities = self._info_num("currentLiabilities")
        receivables = self._info_num("currentReceivables")
        inventory = self._info_num("inventory")
        payables = self._info_num("currentPayables")

        net_working_capital = None
        if current_assets and current_liabilities:
            net_working_capital = current_assets - current_liabilities

        # PP&E
        property_plant_equip = self._info_num("propertyPlantEquipmentNet")

        return {
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "total_debt": total_debt,
            "cash": cash,
            "book_value": book_value,
            "tangible_book_value": tangible_book_value,
            "goodwill": goodwill,
            "intangible_assets": intangible_assets,
            "net_working_capital": net_working_capital,
            "ppe": property_plant_equip,
            "receivables": receivables,
            "inventory": inventory,
            "payables": payables,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "other_liabilities": None  # Could be calculated as total_liabilities - debt - payables
        }

    def get_ppa_adjustments(self, purchase_price: float, premium_percent: float) -> dict:
        """Calculate Purchase Price Allocation adjustments"""
        metrics = self.get_key_metrics()
        bs_metrics = self.get_balance_sheet_metrics()

        # Get book values
        book_value = bs_metrics.get("book_value", 0)
        tangible_book_value = bs_metrics.get("tangible_book_value", book_value)

        if not book_value or book_value <= 0:
            # Estimate from market cap if book value not available
            market_cap = metrics.get("market_cap", 0)
            pe_ratio = metrics.get("pe_ratio", 20)
            if market_cap > 0 and pe_ratio > 0:
                estimated_net_income = market_cap / pe_ratio
                book_value = estimated_net_income * 3  # Rough ROE assumption

        # Calculate goodwill
        goodwill = max(0, purchase_price - tangible_book_value)

        # Write-up/down categories (simplified)
        ppe_write_up = purchase_price * 0.10  # Assume 10% of purchase price goes to PP&E write-up
        intangible_write_up = purchase_price * 0.15  # 15% to identifiable intangibles (brand, customer relationships)

        return {
            "purchase_price": purchase_price,
            "target_book_value": book_value,
            "target_tangible_book_value": tangible_book_value,
            "goodwill": goodwill,
            "ppe_write_up": ppe_write_up,
            "intangible_assets_write_up": intangible_write_up,
            "debt_assumed": bs_metrics.get("total_debt", 0),
            "cash_acquired": bs_metrics.get("cash", 0)
        }

    def get_key_metrics(self) -> dict:
        if not self.success:
            return {"name": self.ticker or "Unknown", "error": self.error}

        revenue    = self._fin_val("revenue")    or self._info_num("totalRevenue")
        ebitda     = self._fin_val("ebitda")
        net_income = self._fin_val("net_income")
        market_cap = self._info_num("marketCap")
        ev         = self._info_num("enterpriseValue")
        ev_ebitda  = self._info_num("enterpriseToEbitda")

        current_price = self._info_num("currentPrice", "regularMarketPrice", "current_price")
        if not current_price and self.source == "yfinance":
            try:
                ticker_obj = yf.Ticker(self.ticker)
                current_price = ticker_obj.fast_info.get("lastPrice", 0)
            except Exception:
                current_price = 0

        ebitda_margin = None
        if ebitda and revenue:
            ebitda_margin = ebitda / revenue
        elif self._info_num("ebitdaMargins"):
            ebitda_margin = self._info_num("ebitdaMargins")

        if not ev_ebitda and ev and ebitda and ebitda > 0:
            ev_ebitda = ev / ebitda

        # Get shares outstanding - CRITICAL FIX
        shares_outstanding = self._info_num("sharesOutstanding")
        if not shares_outstanding or shares_outstanding <= 0:
            # Fallback: estimate from market cap and current price
            if market_cap and market_cap > 0 and current_price and current_price > 0:
                shares_outstanding = market_cap / current_price
            else:
                shares_outstanding = None

        # Get EPS directly - CRITICAL FIX
        eps = self._info_num("trailingEps")
        if not eps and net_income and shares_outstanding and shares_outstanding > 0:
            eps = net_income / shares_outstanding

        return {
            "name":             self.info.get("longName") or self.ticker or "Unknown",
            "sector":           self.info.get("sector", "N/A"),
            "market_cap":       market_cap,
            "enterprise_value": ev,
            "revenue":          revenue,
            "ebitda":           ebitda,
            "net_income":       net_income,
            "ebitda_margin":    ebitda_margin,
            "pe_ratio":         self._info_num("trailingPE"),
            "current_price":    current_price,
            "52w_high":         self._info_num("fiftyTwoWeekHigh"),
            "52w_low":          self._info_num("fiftyTwoWeekLow"),
            "week_52_high":     self._info_num("fiftyTwoWeekHigh"),
            "week_52_low":      self._info_num("fiftyTwoWeekLow"),
            "total_debt":       self._info_num("totalDebt"),
            "cash":             self._info_num("totalCash", "cash"),
            "interest_expense": self._info_num("interestExpense"),
            "shares_outstanding": shares_outstanding,  # ADDED
            "beta":             self._info_num("beta"),
            "eps":              eps,  # ADDED - direct EPS
            "eps_forward":      self._info_num("forwardEps"),
            "ev_ebitda":        ev_ebitda,
            "source":           self.source,
        }

    def get_financial_summary(self) -> pd.DataFrame:
        m  = self.get_key_metrics()
        cur = self.info.get("currency", "")

        # Choose display scale based on currency
        if cur == "INR" or self.ticker.upper().endswith(".NS") or self.ticker.upper().endswith(".BO"):
            sym, scale, unit = "₹", 10_000_000, "Cr"
        else:
            # USD or unknown — display in millions
            sym, scale, unit = "$", 1_000_000, "M"

        def fmt(val):
            if val is None: return "N/A"
            try:    return f"{sym}{float(val)/scale:,.1f} {unit}"
            except: return str(val)

        rows = [
            ("Market Cap",        fmt(m.get("market_cap"))),
            ("Enterprise Value",  fmt(m.get("enterprise_value"))),
            ("Revenue",           fmt(m.get("revenue"))),
            ("EBITDA",            fmt(m.get("ebitda"))),
            ("Net Income",        fmt(m.get("net_income"))),
            ("EBITDA Margin",     f"{m['ebitda_margin']*100:.1f}%" if m.get("ebitda_margin") else "N/A"),
            ("Shares Outstanding", f"{m.get('shares_outstanding', 0)/1e9:.3f}B" if m.get('shares_outstanding') else "N/A"),
            ("EPS (TTM)",         f"₹{m.get('eps', 0):.2f}" if m.get('eps') else "N/A"),
            ("Data Source",       m.get("source", "N/A").upper()),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


# ── Cache wrapper ──────────────────────────────────────────────────────────────

@st.cache_resource(ttl=3600)
def get_company(ticker: str) -> Company:
    return Company(ticker=ticker)