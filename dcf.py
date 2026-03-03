"""
dcf.py — Multi-stage Discounted Cash Flow (DCF) engine.

Model design
────────────
• Stage 1 (years 1–5)  : constant high-growth phase
• Stage 2 (years 6–10) : growth linearly decelerates to the terminal rate
• Terminal value        : Gordon Growth Model on Year-10 FCF
• FCF projection        : revenue-based (projected revenue × FCF margin)
  This is more stable than growing FCF directly for companies whose margins
  are still expanding.

All cash flows are discounted at WACC.  Equity value = EV − debt + cash.
"""

import numpy as np
from config import (
    TERMINAL_GROWTH_RATE, PROJECTION_YEARS, MAX_GROWTH_CAP,
)


# ─── Growth Schedule ──────────────────────────────────────────────────────────

def build_growth_schedule(
    stage1_growth: float,
    terminal_growth: float,
    years: int = PROJECTION_YEARS,
) -> list:
    """
    Build a two-stage annual revenue growth schedule.

    Years 1–5  : stage1_growth (held constant)
    Years 6–10 : linear interpolation from stage1_growth → terminal_growth
    """
    stage1_years = years // 2
    stage2_years = years - stage1_years

    stage1 = [stage1_growth] * stage1_years

    stage2 = []
    for i in range(1, stage2_years + 1):
        frac = i / stage2_years
        rate = stage1_growth * (1 - frac) + terminal_growth * frac
        stage2.append(rate)

    return stage1 + stage2


# ─── FCF Projector ────────────────────────────────────────────────────────────

def project_fcf(
    base_revenue: float,
    fcf_margin: float,
    growth_rates: list,
    fcf_margin_multiplier: float = 1.0,
) -> list:
    """
    Project annual free cash flows using a revenue-growth × margin approach.

    Args:
        base_revenue          : most recent TTM revenue
        fcf_margin            : TTM FCF ÷ revenue
        growth_rates          : per-year revenue growth rates (len = PROJECTION_YEARS)
        fcf_margin_multiplier : bear / bull scenario adjustment

    Returns:
        list of projected FCFs (one per year)
    """
    adjusted_margin = fcf_margin * fcf_margin_multiplier
    projected_fcfs  = []
    current_revenue = base_revenue

    for rate in growth_rates:
        current_revenue *= (1 + rate)
        projected_fcfs.append(current_revenue * adjusted_margin)

    return projected_fcfs


# ─── Core DCF ─────────────────────────────────────────────────────────────────

def run_dcf(
    financials: dict,
    wacc_data: dict,
    manual_growth: float = None,
    terminal_growth: float = TERMINAL_GROWTH_RATE,
    fcf_margin_multiplier: float = 1.0,
) -> dict:
    """
    Execute the DCF model and return a comprehensive result dict.

    Key outputs
    ───────────
    intrinsic_value_per_share  : equity value per diluted share
    enterprise_value           : total firm value (PV FCFs + PV terminal)
    pv_fcfs                    : list of discounted FCFs by year
    terminal_value             : undiscounted Gordon-growth terminal value
    pv_terminal_value          : PV of terminal value
    tv_pct_of_ev               : terminal value as % of total EV
    wacc_used, growth_rate_used, terminal_growth_used, growth_source
    projected_fcfs             : undiscounted year-by-year FCFs

    Returns {"error": <msg>} if critical inputs are missing.
    """
    wacc = wacc_data["wacc"]

    # ── Validate inputs ──────────────────────────────────────────
    base_revenue = financials.get("revenue")
    fcf_margin   = financials.get("fcf_margin")
    shares       = financials.get("shares_outstanding")
    total_debt   = financials.get("total_debt", 0) or 0
    cash         = financials.get("cash",        0) or 0

    if base_revenue is None or base_revenue <= 0:
        return {"error": "Revenue data unavailable — cannot project FCFs."}
    if shares is None or shares <= 0:
        return {"error": "Shares outstanding unavailable — cannot compute per-share value."}

    # ── FCF margin sanity ────────────────────────────────────────
    if fcf_margin is None or fcf_margin < -0.50:
        print("  [Warning] FCF margin missing or severely negative. "
              "Using 5% placeholder — treat DCF output cautiously.")
        fcf_margin  = 0.05
        margin_note = "5% placeholder (data unavailable)"
    elif fcf_margin < 0:
        print(f"  [Warning] Negative FCF margin ({fcf_margin:.1%}). "
              "Company is burning cash — DCF may understate eventual value.")
        margin_note = f"{fcf_margin:.1%} (negative — early stage)"
    else:
        margin_note = f"{fcf_margin:.1%} (TTM)"

    # ── Stage-1 growth rate ──────────────────────────────────────
    if manual_growth is not None:
        stage1_growth = min(max(manual_growth, 0.0), MAX_GROWTH_CAP)
        growth_source = "manual override"
    elif financials.get("revenue_growth") is not None:
        raw = financials["revenue_growth"]
        stage1_growth = min(max(raw, 0.01), MAX_GROWTH_CAP)
        growth_source = f"TTM historical ({raw:.1%} capped at {MAX_GROWTH_CAP:.0%})"
    else:
        stage1_growth = 0.15
        growth_source = "default 15% (no historical data)"

    # ── WACC / terminal-growth conflict guard ────────────────────
    if wacc <= terminal_growth:
        print(f"  [Warning] WACC ({wacc:.2%}) ≤ terminal growth ({terminal_growth:.2%}). "
              "Terminal value undefined — adding 1% buffer to WACC.")
        effective_wacc = terminal_growth + 0.01
    else:
        effective_wacc = wacc

    # ── Build projections ────────────────────────────────────────
    growth_schedule = build_growth_schedule(stage1_growth, terminal_growth)
    projected_fcfs  = project_fcf(
        base_revenue         = base_revenue,
        fcf_margin           = fcf_margin,
        growth_rates         = growth_schedule,
        fcf_margin_multiplier= fcf_margin_multiplier,
    )

    # ── Discount FCFs to PV ──────────────────────────────────────
    pv_fcfs = [fcf / (1 + effective_wacc) ** t
               for t, fcf in enumerate(projected_fcfs, start=1)]

    # ── Terminal value (Gordon Growth on Year-10 FCF) ────────────
    terminal_value    = projected_fcfs[-1] * (1 + terminal_growth) / (effective_wacc - terminal_growth)
    pv_terminal_value = terminal_value / (1 + effective_wacc) ** PROJECTION_YEARS

    # ── Aggregate ────────────────────────────────────────────────
    enterprise_value = sum(pv_fcfs) + pv_terminal_value
    equity_value     = enterprise_value - total_debt + cash

    intrinsic_value_per_share = equity_value / shares

    tv_pct = pv_terminal_value / enterprise_value if enterprise_value != 0 else None

    return {
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "enterprise_value":          enterprise_value,
        "equity_value":              equity_value,
        "pv_fcfs":                   pv_fcfs,
        "terminal_value":            terminal_value,
        "pv_terminal_value":         pv_terminal_value,
        "tv_pct_of_ev":              tv_pct,
        "wacc_used":                 effective_wacc,
        "growth_rate_used":          stage1_growth,
        "growth_source":             growth_source,
        "terminal_growth_used":      terminal_growth,
        "fcf_margin_note":           margin_note,
        "projected_fcfs":            projected_fcfs,
        "growth_schedule":           growth_schedule,
    }


# ─── Public Summary ───────────────────────────────────────────────────────────

def dcf_summary(
    financials: dict,
    market_data: dict,
    wacc_data: dict,
    manual_growth: float = None,
) -> dict:
    """
    Wrapper: run DCF and append current-price context (upside %, verdict).

    Verdict thresholds:
        > +10%  → UNDERVALUED
        < −10%  → OVERVALUED
        else    → FAIRLY VALUED
    """
    result = run_dcf(financials, wacc_data, manual_growth=manual_growth)

    if "error" in result:
        return result

    current_price = market_data.get("price")
    if current_price and current_price > 0:
        iv      = result["intrinsic_value_per_share"]
        upside  = (iv - current_price) / current_price
        verdict = (
            "UNDERVALUED"  if upside >  0.10 else
            "OVERVALUED"   if upside < -0.10 else
            "FAIRLY VALUED"
        )
        result["current_price"] = current_price
        result["upside_pct"]    = upside
        result["verdict"]       = verdict

    return result
