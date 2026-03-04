"""
utils.py — Data fetching, WACC construction, and shared helpers.

Primary source: yahooquery — hits different Yahoo Finance endpoints that
are not blocked on cloud server IPs (Streamlit Cloud / AWS).
Fallback: cached_data.json — used only when the live fetch fails.
"""

import json
import math
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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


def _safe_num(val, fallback=None):
    """Return val if it's a usable number, else fallback."""
    try:
        if val is None:
            return fallback
        f = float(val)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback


def _yq_fetch(ticker: str):
    """
    Fetch all needed data from yahooquery in one call.
    Returns (financial_data, price_data, summary_detail, key_stats) dicts,
    all normalised so missing/error responses become empty dicts.
    """
    from yahooquery import Ticker
    sym = ticker.upper()
    yq  = Ticker(sym)

    def _safe(d):
        v = d.get(sym, {})
        return v if isinstance(v, dict) else {}

    return (
        _safe(yq.financial_data),
        _safe(yq.price),
        _safe(yq.summary_detail),
        _safe(yq.key_stats),
        yq,          # keep the Ticker object for statement calls
    )


# ── yfinance-compatible wrapper used by comps.py ──────────────────────────────

class _YQTicker:
    """
    Thin yahooquery wrapper with a .info property that looks like yfinance's
    info dict.  comps.py calls get_ticker(t).info — this satisfies that.
    """
    def __init__(self, symbol: str):
        self._sym   = symbol.upper()
        self._cache = None

    @property
    def info(self) -> dict:
        if self._cache is not None:
            return self._cache
        try:
            fd, pr, sd, ks, _ = _yq_fetch(self._sym)
            price = _safe_num(pr.get("regularMarketPrice"))
            if not price:
                self._cache = {}
                return self._cache

            market_cap = _safe_num(pr.get("marketCap"))
            total_debt = _safe_num(fd.get("totalDebt"), 0)
            cash       = _safe_num(fd.get("totalCash"),  0)
            ev         = (market_cap + total_debt - cash) if market_cap else None

            self._cache = {
                "currentPrice":                   price,
                "regularMarketPrice":             price,
                "marketCap":                      market_cap,
                "totalDebt":                      total_debt,
                "totalCash":                      cash,
                "enterpriseValue":                ev,
                "ebitda":                         _safe_num(fd.get("ebitda")),
                "totalRevenue":                   _safe_num(fd.get("totalRevenue")),
                "revenueGrowth":                  _safe_num(fd.get("revenueGrowth")),
                "netIncomeToCommon":              _safe_num(fd.get("netIncomeToCommon")),
                "freeCashflow":                   _safe_num(fd.get("freeCashflow")),
                "operatingCashflow":              _safe_num(fd.get("operatingCashflow")),
                "beta":                           _safe_num(sd.get("beta")),
                "sharesOutstanding":              _safe_num(ks.get("sharesOutstanding")),
                "trailingPE":                     _safe_num(sd.get("trailingPE")),
                "forwardPE":                      _safe_num(sd.get("forwardPE")),
                "trailingEps":                    _safe_num(ks.get("trailingEps")),
                "epsTrailingTwelveMonths":        _safe_num(ks.get("trailingEps")),
                "priceToSalesTrailing12Months":   _safe_num(sd.get("priceToSalesTrailing12Months")),
                "pegRatio":                       _safe_num(ks.get("pegRatio")),
                "enterpriseToEbitda":             _safe_num(ks.get("enterpriseToEbitda")),
            }
        except Exception:
            self._cache = {}
        return self._cache


# ─── Financial Data ───────────────────────────────────────────────────────────

def fetch_financials(ticker: str) -> dict:
    """
    Pull financials via yahooquery.
    Falls back to cached_data.json if the live fetch fails.
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
        fd, pr, sd, ks, yq = _yq_fetch(ticker)

        price = _safe_num(pr.get("regularMarketPrice"))
        if not price:
            raise ValueError("No price from yahooquery — ticker may be invalid")

        result["ebitda"]             = _safe_num(fd.get("ebitda"))
        result["net_income"]         = _safe_num(fd.get("netIncomeToCommon"))
        result["total_debt"]         = _safe_num(fd.get("totalDebt"), 0)
        result["cash"]               = _safe_num(fd.get("totalCash"),  0)
        result["shares_outstanding"] = _safe_num(ks.get("sharesOutstanding"))
        result["beta"]               = _safe_num(sd.get("beta"))
        result["revenue"]            = _safe_num(fd.get("totalRevenue"))
        result["revenue_growth"]     = _safe_num(fd.get("revenueGrowth"))
        result["fcf"]                = _safe_num(fd.get("freeCashflow"))

        # EBIT + revenue growth from income statement (more precise)
        try:
            sym = ticker.upper()
            is_df = yq.income_statement(frequency="annual", trailing=True)
            if is_df is not None and not isinstance(is_df, str) and not is_df.empty:
                if "asOfDate" in is_df.columns:
                    is_df = is_df.sort_values("asOfDate", ascending=False)
                row = is_df.iloc[0]
                for col in ("EBIT", "OperatingIncome"):
                    if col in is_df.columns:
                        v = _safe_num(row.get(col))
                        if v is not None:
                            result["ebit"] = v
                            break
                if result["revenue"] is None and "TotalRevenue" in is_df.columns:
                    result["revenue"] = _safe_num(row.get("TotalRevenue"))
                if result["revenue_growth"] is None and "TotalRevenue" in is_df.columns and len(is_df) >= 2:
                    rv = _safe_num(is_df.iloc[0]["TotalRevenue"])
                    rp = _safe_num(is_df.iloc[1]["TotalRevenue"])
                    if rv and rp and rp != 0:
                        result["revenue_growth"] = (rv - rp) / abs(rp)
        except Exception:
            pass

        # FCF from cashflow statement if freeCashflow missing
        if result["fcf"] is None:
            try:
                cf_df = yq.cash_flow(frequency="annual", trailing=True)
                if cf_df is not None and not isinstance(cf_df, str) and not cf_df.empty:
                    if "asOfDate" in cf_df.columns:
                        cf_df = cf_df.sort_values("asOfDate", ascending=False)
                    row = cf_df.iloc[0]
                    op_cf = _safe_num(row.get("OperatingCashFlow"))
                    capex = _safe_num(row.get("CapitalExpenditure"), 0)
                    if op_cf is not None:
                        result["fcf"] = op_cf - abs(capex)
            except Exception:
                pass

        rev, fcf = result["revenue"], result["fcf"]
        if rev and rev > 0 and fcf is not None:
            result["fcf_margin"] = fcf / rev

        return result

    except Exception as exc:
        print(f"  [Info] yahooquery fetch failed for {ticker} ({exc}). Using cache.")

    cached = _cached(ticker, "financials")
    if cached:
        cached["_from_cache"] = True
        return cached

    result["_from_cache"] = True
    return result


# ─── WACC Builder ─────────────────────────────────────────────────────────────

def build_wacc(ticker: str, manual_wacc: float = None) -> dict:
    """Compute WACC via CAPM."""
    if manual_wacc is not None:
        return {
            "wacc": manual_wacc, "cost_of_equity": None,
            "cost_of_debt": None, "debt_weight": None,
            "equity_weight": None, "beta": None, "manual_override": True,
        }

    fin = fetch_financials(ticker)
    md  = get_market_data(ticker)

    beta       = fin.get("beta") or DEFAULT_BETA
    total_debt = fin.get("total_debt") or md.get("total_debt") or 0
    market_cap = md.get("market_cap") or 0

    cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM
    cost_of_debt   = DEFAULT_DEBT_COST

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
    """Fetch current price, market cap, and EV via yahooquery."""
    blank = {
        "ticker": ticker, "price": None, "market_cap": None,
        "ev": None, "total_debt": 0, "cash": 0, "_from_cache": False,
    }
    try:
        fd, pr, sd, ks, _ = _yq_fetch(ticker)

        price = _safe_num(pr.get("regularMarketPrice"))
        if not price:
            raise ValueError("No price returned")

        market_cap = _safe_num(pr.get("marketCap"))
        total_debt = _safe_num(fd.get("totalDebt"), 0)
        cash       = _safe_num(fd.get("totalCash"),  0)
        ev         = (market_cap + total_debt - cash) if market_cap else None

        return {
            "ticker": ticker, "price": price, "market_cap": market_cap,
            "ev": ev, "total_debt": total_debt, "cash": cash, "_from_cache": False,
        }

    except Exception as exc:
        print(f"  [Info] yahooquery market data failed for {ticker} ({exc}). Using cache.")

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


def get_ticker(symbol: str) -> _YQTicker:
    """Returns a yahooquery-backed ticker with a yfinance-compatible .info property."""
    return _YQTicker(symbol)


def get_cached_tickers() -> list:
    """Return sorted list of tickers available in the local cache."""
    return sorted(_CACHE.keys())
