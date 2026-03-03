"""
comps.py — Comparable Company Analysis (Comps) engine.

For each company in the peer group the engine fetches live multiples:
    EV/EBITDA  — enterprise-value-based; best for capital-structure comparisons
    P/E        — price-to-earnings (trailing)
    P/S        — price-to-sales; useful for pre-profit names
    PEG        — P/E divided by earnings-growth rate; growth-adjusted P/E proxy

Sector medians are applied to the target company to derive implied share prices —
the standard "trading comps" methodology used in sell-side equity research.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from config import DEFAULT_PEER_GROUP
from utils import safe_divide, get_ticker, _CACHE


# ─── Single-Ticker Metrics ────────────────────────────────────────────────────

def fetch_comp_metrics(ticker: str) -> dict:
    """
    Pull current multiples for one ticker.
    Returns a dict; any unavailable field is None (never raises).
    """
    blank = {
        "ticker": ticker, "price": None, "market_cap": None, "ev": None,
        "ebitda": None, "revenue": None,
        "ev_ebitda": None, "pe_ratio": None, "ps_ratio": None, "peg_ratio": None,
        "revenue_growth": None,
    }
    try:
        info = get_ticker(ticker).info
        # If Yahoo blocked the request, info will be empty — fall through to cache
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            raise ValueError("Blocked")

        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        total_debt = info.get("totalDebt", 0) or 0
        cash       = info.get("totalCash",  0) or 0
        ebitda     = info.get("ebitda")
        revenue    = info.get("totalRevenue")
        rev_growth = info.get("revenueGrowth")   # decimal (e.g. 0.22 = 22%)

        ev = (market_cap + total_debt - cash) if market_cap else None

        # EV/EBITDA — prefer yfinance's own field; compute if missing
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda is None or ev_ebitda <= 0:
            ev_ebitda = safe_divide(ev, ebitda) if (ev and ebitda and ebitda > 0) else None

        # P/E — trailing preferred; forward as fallback
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        if pe_ratio and pe_ratio <= 0:
            pe_ratio = None

        # P/S — market cap ÷ revenue
        ps_ratio = info.get("priceToSalesTrailing12Months")
        if ps_ratio is None:
            ps_ratio = safe_divide(market_cap, revenue) if (market_cap and revenue and revenue > 0) else None

        # PEG — yfinance often provides this directly
        peg_ratio = info.get("pegRatio")
        # Fallback: P/E ÷ (revenue growth × 100)  — approximate
        if peg_ratio is None and pe_ratio and rev_growth and rev_growth > 0:
            peg_ratio = pe_ratio / (rev_growth * 100)
        if peg_ratio and peg_ratio <= 0:
            peg_ratio = None

        return {
            "ticker":         ticker,
            "price":          price,
            "market_cap":     market_cap,
            "ev":             ev,
            "ebitda":         ebitda,
            "revenue":        revenue,
            "ev_ebitda":      ev_ebitda,
            "pe_ratio":       pe_ratio,
            "ps_ratio":       ps_ratio,
            "peg_ratio":      peg_ratio,
            "revenue_growth": rev_growth,
        }

    except Exception as exc:
        print(f"  [Info] Live comps fetch failed for {ticker} ({exc}). Using cache.")
        cached = _CACHE.get(ticker.upper(), {}).get("comp_metrics")
        if cached:
            return cached
        return blank


# ─── Peer-Group Analysis ──────────────────────────────────────────────────────

def run_comps(
    target_ticker: str,
    peers: list = None,
    market_data: dict = None,
) -> dict:
    """
    Run the full comparable-company analysis.

    Steps:
        1. Fetch multiples for all peers + target
        2. Build a ranked DataFrame (sorted by EV/EBITDA descending)
        3. Compute sector medians
        4. Apply medians to the target to derive implied share prices

    Returns dict with:
        comps_df       : DataFrame of all tickers × multiples
        implied_prices : {method: implied_price} from median multiples
        sector_medians : {multiple: median_value}
        target_ticker  : echoed for convenience
    """
    if peers is None:
        peers = [t for t in DEFAULT_PEER_GROUP if t != target_ticker]

    # Deduplicate and ensure target is first
    all_tickers = [target_ticker] + [p for p in peers if p != target_ticker]

    print(f"\n  Fetching comps data for {len(all_tickers)} companies …")
    records = []
    for t in all_tickers:
        print(f"    {t} … ", end="", flush=True)
        rec = fetch_comp_metrics(t)
        records.append(rec)
        print("done")

    df = pd.DataFrame(records).set_index("ticker")

    # ── Sector medians (all peers including target) ──────────────
    def _median(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s[s > 0].median()   # exclude negative / zero multiples

    medians = {
        "ev_ebitda": _median("ev_ebitda"),
        "pe_ratio":  _median("pe_ratio"),
        "ps_ratio":  _median("ps_ratio"),
        "peg_ratio": _median("peg_ratio"),
    }

    # ── Implied prices for the target ────────────────────────────
    implied_prices: dict = {}

    target_row   = df.loc[target_ticker] if target_ticker in df.index else None
    target_price = market_data.get("price")     if market_data else None
    target_mcap  = market_data.get("market_cap") if market_data else None
    target_debt  = (market_data.get("total_debt", 0) or 0) if market_data else 0
    target_cash  = (market_data.get("cash",  0) or 0)       if market_data else 0

    if target_row is not None and target_price and target_mcap and target_price > 0:
        target_shares = target_mcap / target_price  # implied shares

        # — EV/EBITDA implied price —
        t_ebitda = target_row.get("ebitda")
        if (medians["ev_ebitda"] and t_ebitda and t_ebitda > 0
                and not np.isnan(medians["ev_ebitda"])):
            implied_ev     = medians["ev_ebitda"] * t_ebitda
            implied_equity = implied_ev - target_debt + target_cash
            implied_prices["ev_ebitda"] = implied_equity / target_shares

        # — P/S implied price —
        t_revenue = target_row.get("revenue")
        if (medians["ps_ratio"] and t_revenue and t_revenue > 0
                and not np.isnan(medians["ps_ratio"])):
            implied_prices["ps_ratio"] = (medians["ps_ratio"] * t_revenue) / target_shares

        # — P/E implied price —
        try:
            eps = get_ticker(target_ticker).info.get("trailingEps") or \
                  get_ticker(target_ticker).info.get("epsTrailingTwelveMonths")
            if (eps and eps > 0 and medians["pe_ratio"]
                    and not np.isnan(medians["pe_ratio"])):
                implied_prices["pe_ratio"] = medians["pe_ratio"] * eps
        except Exception:
            pass

    # Sort by EV/EBITDA descending for the output table
    df_sorted = df.sort_values("ev_ebitda", ascending=False, na_position="last")

    return {
        "comps_df":       df_sorted,
        "implied_prices": implied_prices,
        "sector_medians": medians,
        "target_ticker":  target_ticker,
    }


# ─── Public Summary ───────────────────────────────────────────────────────────

def comps_summary(
    target_ticker: str,
    peers: list = None,
    financials: dict = None,   # kept for signature consistency; not used directly
    market_data: dict = None,
) -> dict:
    """Thin wrapper — callers use this rather than run_comps directly."""
    return run_comps(target_ticker, peers=peers, market_data=market_data)
