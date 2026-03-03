"""
utils.py — Data fetching, WACC construction, and shared helpers.

All external I/O is concentrated here so the rest of the codebase stays clean.
yfinance is the only data source; no API keys are required.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM,
    DEFAULT_BETA, DEFAULT_DEBT_COST, DEFAULT_TAX_RATE,
)


# ─── Financial Data ───────────────────────────────────────────────────────────

def fetch_financials(ticker: str) -> dict:
    """
    Pull key income-statement, balance-sheet, and cash-flow data via yfinance.

    Returns a dict with:
        ticker, revenue, ebit, ebitda, net_income, fcf, total_debt, cash,
        shares_outstanding, beta, revenue_growth, fcf_margin
    Missing fields are None; warnings are printed to stdout.
    """
    stock = yf.Ticker(ticker)

    result: dict = {
        "ticker":              ticker,
        "revenue":             None,
        "ebit":                None,
        "ebitda":              None,
        "net_income":          None,
        "fcf":                 None,
        "total_debt":          None,
        "cash":                None,
        "shares_outstanding":  None,
        "beta":                None,
        "revenue_growth":      None,
        "fcf_margin":          None,
    }

    # ── 1. info dict (fast) ──────────────────────────────────────
    try:
        info = stock.info
        result["ebitda"]            = info.get("ebitda")
        result["net_income"]        = info.get("netIncomeToCommon")
        result["total_debt"]        = info.get("totalDebt",  0) or 0
        result["cash"]              = info.get("totalCash",  0) or 0
        result["shares_outstanding"]= info.get("sharesOutstanding")
        result["beta"]              = info.get("beta")
        result["revenue"]           = info.get("totalRevenue")
        # Revenue growth (YoY) — provided by yfinance when available
        if info.get("revenueGrowth") is not None:
            result["revenue_growth"] = info["revenueGrowth"]
    except Exception as exc:
        print(f"  [Warning] Could not fetch .info for {ticker}: {exc}")

    # ── 2. Income statement ──────────────────────────────────────
    try:
        inc = stock.income_stmt
        if inc is not None and not inc.empty:
            col = inc.columns[0]   # most recent annual period

            # EBIT / Operating Income
            for label in ("EBIT", "Operating Income", "Ebit"):
                if label in inc.index:
                    result["ebit"] = float(inc.loc[label, col])
                    break

            # Revenue fallback if info dict was missing it
            if result["revenue"] is None:
                for label in ("Total Revenue", "Revenue"):
                    if label in inc.index:
                        result["revenue"] = float(inc.loc[label, col])
                        break

            # Revenue growth from two annual periods if still missing
            if result["revenue_growth"] is None and inc.shape[1] >= 2:
                col_prev = inc.columns[1]
                for label in ("Total Revenue", "Revenue"):
                    if label in inc.index:
                        rev_now  = inc.loc[label, col]
                        rev_prev = inc.loc[label, col_prev]
                        if rev_prev and rev_prev != 0:
                            result["revenue_growth"] = (rev_now - rev_prev) / abs(rev_prev)
                        break
    except Exception as exc:
        print(f"  [Warning] Could not parse income statement for {ticker}: {exc}")

    # ── 3. Cash-flow statement ───────────────────────────────────
    try:
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            col = cf.columns[0]

            operating_cf = None
            capex = 0.0

            for label in ("Operating Cash Flow", "Cash Flow From Operations",
                          "Total Cash From Operating Activities",
                          "Net Cash Provided By Operating Activities"):
                if label in cf.index:
                    operating_cf = float(cf.loc[label, col])
                    break

            for label in ("Capital Expenditure", "Capital Expenditures",
                          "Purchase Of PPE", "Purchases Of Property Plant And Equipment"):
                if label in cf.index:
                    capex = float(cf.loc[label, col])
                    break

            if operating_cf is not None:
                # CapEx is usually reported as a negative number in cash-flow statements
                result["fcf"] = operating_cf - abs(capex)
    except Exception as exc:
        print(f"  [Warning] Could not parse cash flow for {ticker}: {exc}")

    # ── 4. Derived metrics ───────────────────────────────────────
    rev = result["revenue"]
    fcf = result["fcf"]
    if rev and rev > 0 and fcf is not None:
        result["fcf_margin"] = fcf / rev

    # ── 5. Warn on critical gaps ─────────────────────────────────
    for key in ("revenue", "fcf", "shares_outstanding"):
        if result[key] is None:
            print(f"  [Warning] {ticker}: missing '{key}' — some outputs may be N/A")

    return result


# ─── WACC Builder ─────────────────────────────────────────────────────────────

def build_wacc(ticker: str, manual_wacc: float = None) -> dict:
    """
    Compute Weighted Average Cost of Capital (WACC).

    Cost of equity   → CAPM: Rf + β × ERP
    Cost of debt     → interest expense / total debt (or default 5%)
    Weights          → market cap and total debt (book value)

    If manual_wacc is provided, all computation is skipped.

    Returns dict with keys:
        wacc, cost_of_equity, cost_of_debt, debt_weight, equity_weight,
        beta, manual_override
    """
    if manual_wacc is not None:
        return {
            "wacc": manual_wacc,
            "cost_of_equity": None,
            "cost_of_debt":   None,
            "debt_weight":    None,
            "equity_weight":  None,
            "beta":           None,
            "manual_override": True,
        }

    try:
        stock  = yf.Ticker(ticker)
        info   = stock.info

        beta        = info.get("beta") or DEFAULT_BETA
        total_debt  = info.get("totalDebt", 0) or 0
        market_cap  = info.get("marketCap", 0) or 0

        # Cost of equity — CAPM
        cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

        # Cost of debt — interest expense ÷ total debt where available
        cost_of_debt = DEFAULT_DEBT_COST
        try:
            inc = stock.income_stmt
            if inc is not None and not inc.empty and total_debt > 0:
                col = inc.columns[0]
                for label in ("Interest Expense", "Interest Expense Non Operating",
                              "Net Interest Income"):
                    if label in inc.index:
                        interest = abs(float(inc.loc[label, col]))
                        if interest > 0:
                            implied = interest / total_debt
                            # Clamp to a plausible range (2% – 15%)
                            cost_of_debt = max(0.02, min(0.15, implied))
                        break
        except Exception:
            pass   # stick with DEFAULT_DEBT_COST

        # Capital-structure weights
        total_capital = market_cap + total_debt
        if total_capital <= 0:
            equity_weight, debt_weight = 1.0, 0.0
        else:
            equity_weight = market_cap  / total_capital
            debt_weight   = total_debt  / total_capital

        after_tax_debt = cost_of_debt * (1 - DEFAULT_TAX_RATE)
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_debt

        return {
            "wacc":            wacc,
            "cost_of_equity":  cost_of_equity,
            "cost_of_debt":    cost_of_debt,
            "debt_weight":     debt_weight,
            "equity_weight":   equity_weight,
            "beta":            beta,
            "manual_override": False,
        }

    except Exception as exc:
        print(f"  [Warning] WACC build failed for {ticker}: {exc}. Using CAPM with default β.")
        cost_of_equity = RISK_FREE_RATE + DEFAULT_BETA * EQUITY_RISK_PREMIUM
        return {
            "wacc":            cost_of_equity,
            "cost_of_equity":  cost_of_equity,
            "cost_of_debt":    DEFAULT_DEBT_COST,
            "debt_weight":     0.0,
            "equity_weight":   1.0,
            "beta":            DEFAULT_BETA,
            "manual_override": False,
        }


# ─── Market Data ──────────────────────────────────────────────────────────────

def get_market_data(ticker: str) -> dict:
    """
    Fetch current price, market cap, and enterprise value.
    EV = Market Cap + Total Debt − Cash & Equivalents.
    """
    try:
        info       = yf.Ticker(ticker).info
        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        total_debt = info.get("totalDebt", 0) or 0
        cash       = info.get("totalCash",  0) or 0
        ev         = (market_cap + total_debt - cash) if market_cap else None

        return {
            "ticker":     ticker,
            "price":      price,
            "market_cap": market_cap,
            "ev":         ev,
            "total_debt": total_debt,
            "cash":       cash,
        }

    except Exception as exc:
        print(f"  [Error] Market data fetch failed for {ticker}: {exc}")
        return {
            "ticker": ticker, "price": None, "market_cap": None,
            "ev": None, "total_debt": 0, "cash": 0,
        }


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def format_large_number(n, decimals: int = 2) -> str:
    """Render a raw number as $1.23T / $456.78B / $789.01M / $1.23."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    sign  = "-" if n < 0 else ""
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{sign}${abs_n/1e12:.{decimals}f}T"
    if abs_n >= 1e9:
        return f"{sign}${abs_n/1e9:.{decimals}f}B"
    if abs_n >= 1e6:
        return f"{sign}${abs_n/1e6:.{decimals}f}M"
    return f"{sign}${abs_n:.{decimals}f}"


def safe_divide(numerator, denominator, fallback=None):
    """Division that returns fallback instead of raising on zero / None."""
    try:
        if denominator is None or denominator == 0:
            return fallback
        return numerator / denominator
    except Exception:
        return fallback
