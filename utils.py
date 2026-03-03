"""
utils.py — Data fetching, WACC construction, and shared helpers.

Yahoo Finance aggressively blocks cloud server IPs (Streamlit Cloud / AWS).
Strategy: attempt a live yfinance fetch; if the price comes back empty,
fall back to cached_data.json which was pre-fetched on a clean local connection.
The dashboard shows a banner whenever cached data is in use.
"""

import json
import math
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM,
    DEFAULT_BETA, DEFAULT_DEBT_COST, DEFAULT_TAX_RATE,
)

# ── Load fallback cache ───────────────────────────────────────────────────────
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_data.json")
try:
    with open(_CACHE_PATH) as _f:
        _CACHE: dict = json.load(_f)
except Exception:
    _CACHE = {}


def _cached(ticker: str, key: str):
    """Return cached sub-dict for a ticker, or empty dict if missing."""
    return _CACHE.get(ticker.upper(), {}).get(key, {})


# ─── Financial Data ───────────────────────────────────────────────────────────

def fetch_financials(ticker: str) -> dict:
    """
    Pull income-statement, balance-sheet, and cash-flow data via yfinance.
    Falls back to cached_data.json when the live fetch is blocked.

    Returns a dict with:
        ticker, revenue, ebit, ebitda, net_income, fcf, total_debt, cash,
        shares_outstanding, beta, revenue_growth, fcf_margin, _from_cache
    """
    result: dict = {
        "ticker":             ticker,
        "revenue":            None,
        "ebit":               None,
        "ebitda":             None,
        "net_income":         None,
        "fcf":                None,
        "total_debt":         None,
        "cash":               None,
        "shares_outstanding": None,
        "beta":               None,
        "revenue_growth":     None,
        "fcf_margin":         None,
        "_from_cache":        False,
    }

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # Treat an empty or price-less info dict as a failed fetch
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            raise ValueError("Empty info — likely blocked by Yahoo Finance")

        result["ebitda"]             = info.get("ebitda")
        result["net_income"]         = info.get("netIncomeToCommon")
        result["total_debt"]         = info.get("totalDebt",  0) or 0
        result["cash"]               = info.get("totalCash",  0) or 0
        result["shares_outstanding"] = info.get("sharesOutstanding")
        result["beta"]               = info.get("beta")
        result["revenue"]            = info.get("totalRevenue")
        if info.get("revenueGrowth") is not None:
            result["revenue_growth"] = info["revenueGrowth"]

        # Income statement
        try:
            inc = stock.income_stmt
            if inc is not None and not inc.empty:
                col = inc.columns[0]
                for label in ("EBIT", "Operating Income", "Ebit"):
                    if label in inc.index:
                        result["ebit"] = float(inc.loc[label, col])
                        break
                if result["revenue"] is None:
                    for label in ("Total Revenue", "Revenue"):
                        if label in inc.index:
                            result["revenue"] = float(inc.loc[label, col])
                            break
                if result["revenue_growth"] is None and inc.shape[1] >= 2:
                    col_prev = inc.columns[1]
                    for label in ("Total Revenue", "Revenue"):
                        if label in inc.index:
                            rv, rp = inc.loc[label, col], inc.loc[label, col_prev]
                            if rp and rp != 0:
                                result["revenue_growth"] = (rv - rp) / abs(rp)
                            break
        except Exception:
            pass

        # Cash-flow statement
        try:
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                col = cf.columns[0]
                op_cf, capex = None, 0.0
                for label in ("Operating Cash Flow", "Cash Flow From Operations",
                              "Total Cash From Operating Activities"):
                    if label in cf.index:
                        op_cf = float(cf.loc[label, col])
                        break
                for label in ("Capital Expenditure", "Capital Expenditures",
                              "Purchase Of PPE"):
                    if label in cf.index:
                        capex = float(cf.loc[label, col])
                        break
                if op_cf is not None:
                    result["fcf"] = op_cf - abs(capex)
        except Exception:
            pass

        # FCF margin
        rev, fcf = result["revenue"], result["fcf"]
        if rev and rev > 0 and fcf is not None:
            result["fcf_margin"] = fcf / rev

        return result

    except Exception as exc:
        print(f"  [Info] Live fetch failed for {ticker} ({exc}). Using cache.")

    # ── Fallback to cache ─────────────────────────────────────────
    cached = _cached(ticker, "financials")
    if cached:
        cached["_from_cache"] = True
        return cached

    result["_from_cache"] = True
    return result


# ─── WACC Builder ─────────────────────────────────────────────────────────────

def build_wacc(ticker: str, manual_wacc: float = None) -> dict:
    """
    Compute WACC via CAPM. Uses cached financials if live fetch is unavailable.
    """
    if manual_wacc is not None:
        return {
            "wacc": manual_wacc, "cost_of_equity": None,
            "cost_of_debt": None, "debt_weight": None,
            "equity_weight": None, "beta": None, "manual_override": True,
        }

    # Pull beta + capital structure from financials (already cache-aware)
    fin = fetch_financials(ticker)
    md  = get_market_data(ticker)

    beta       = fin.get("beta") or DEFAULT_BETA
    total_debt = fin.get("total_debt") or md.get("total_debt") or 0
    market_cap = md.get("market_cap") or 0

    cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM
    cost_of_debt   = DEFAULT_DEBT_COST   # simplified; interest-expense parse is optional

    total_capital = market_cap + total_debt
    if total_capital <= 0:
        equity_weight, debt_weight = 1.0, 0.0
    else:
        equity_weight = market_cap  / total_capital
        debt_weight   = total_debt  / total_capital

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - DEFAULT_TAX_RATE)

    return {
        "wacc":            wacc,
        "cost_of_equity":  cost_of_equity,
        "cost_of_debt":    cost_of_debt,
        "debt_weight":     debt_weight,
        "equity_weight":   equity_weight,
        "beta":            beta,
        "manual_override": False,
    }


# ─── Market Data ──────────────────────────────────────────────────────────────

def get_market_data(ticker: str) -> dict:
    """
    Fetch current price, market cap, and EV.
    Falls back to cached_data.json when blocked.
    """
    blank = {
        "ticker": ticker, "price": None, "market_cap": None,
        "ev": None, "total_debt": 0, "cash": 0, "_from_cache": False,
    }
    try:
        info       = yf.Ticker(ticker).info or {}
        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")

        if not price:
            raise ValueError("No price returned — likely blocked")

        total_debt = info.get("totalDebt", 0) or 0
        cash       = info.get("totalCash",  0) or 0
        ev         = (market_cap + total_debt - cash) if market_cap else None

        return {
            "ticker": ticker, "price": price, "market_cap": market_cap,
            "ev": ev, "total_debt": total_debt, "cash": cash, "_from_cache": False,
        }

    except Exception as exc:
        print(f"  [Info] Live market data failed for {ticker} ({exc}). Using cache.")

    cached = _cached(ticker, "market_data")
    if cached:
        cached["_from_cache"] = True
        return cached

    blank["_from_cache"] = True
    return blank


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def format_large_number(n, decimals: int = 2) -> str:
    """Render a raw number as $1.23T / $456.78B / $789.01M."""
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "N/A"
    sign, abs_n = ("-" if n < 0 else ""), abs(n)
    if abs_n >= 1e12: return f"{sign}${abs_n/1e12:.{decimals}f}T"
    if abs_n >= 1e9:  return f"{sign}${abs_n/1e9:.{decimals}f}B"
    if abs_n >= 1e6:  return f"{sign}${abs_n/1e6:.{decimals}f}M"
    return f"{sign}${abs_n:.{decimals}f}"


def safe_divide(numerator, denominator, fallback=None):
    try:
        return fallback if (denominator is None or denominator == 0) else numerator / denominator
    except Exception:
        return fallback


def get_ticker(symbol: str) -> yf.Ticker:
    """Plain yf.Ticker — kept for backward compatibility with comps.py."""
    return yf.Ticker(symbol)
