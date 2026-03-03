"""
sensitivity.py — Sensitivity analysis and scenario modelling.

Two outputs:
  1. WACC × terminal-growth-rate matrix (5 × 5 grid of intrinsic values)
  2. Bear / Base / Bull scenario table with explicit assumption sets
"""

import numpy as np
import pandas as pd

from config import (
    WACC_BPS_STEPS, TERMINAL_GROWTH_RANGE,
    BEAR_FCF_MARGIN_COMPRESSION, BASE_FCF_MARGIN_MULTIPLIER, BULL_FCF_MARGIN_EXPANSION,
    MAX_GROWTH_CAP,
)
from dcf import run_dcf


# ─── Sensitivity Matrix ───────────────────────────────────────────────────────

def build_sensitivity_matrix(
    financials: dict,
    market_data: dict,
    wacc_data: dict,
    wacc_range: list = None,
    tg_range: list = None,
) -> pd.DataFrame:
    """
    Compute intrinsic value per share for every combination of WACC and
    terminal growth rate.

    Rows   → WACC values (base ± 200 bps)
    Columns → terminal growth rates (1 % – 5 %)
    Cells  → implied intrinsic value per share

    Returns a DataFrame styled for tabulate display and heatmap rendering.
    """
    base_wacc = wacc_data["wacc"]

    if wacc_range is None:
        wacc_range = [base_wacc + bps / 10_000 for bps in WACC_BPS_STEPS]
    if tg_range is None:
        tg_range = TERMINAL_GROWTH_RANGE

    matrix: dict = {}

    for tg in tg_range:
        col: dict = {}
        for w in wacc_range:
            temp_wacc = {**wacc_data, "wacc": max(w, 0.01)}   # floor at 1%
            result    = run_dcf(financials, temp_wacc, terminal_growth=tg)
            col[f"{w:.2%}"] = (
                result.get("intrinsic_value_per_share")
                if "error" not in result else np.nan
            )
        matrix[f"{tg:.0%}"] = col

    df = pd.DataFrame(matrix)
    df.index.name   = "WACC \\ Tgrowth"
    df.columns.name = "Terminal Growth"
    return df


# ─── Scenario Analysis ────────────────────────────────────────────────────────

def build_scenarios(
    financials: dict,
    market_data: dict,
    wacc_data: dict,
    manual_growth: float = None,
) -> dict:
    """
    Build three named scenarios with explicit, defensible assumption sets.

    Bear — adverse macro: higher discount rates, slower growth, margin contraction
    Base — central estimate: as-calculated WACC, TTM/analyst growth
    Bull — optimistic: lower rates, accelerating growth, margin expansion

    Each scenario returns:
        wacc, growth, terminal_growth, fcf_margin_mult,
        intrinsic_value (per share), upside_pct vs. current price
    """
    base_wacc   = wacc_data["wacc"]
    base_growth = manual_growth or financials.get("revenue_growth") or 0.15
    # Clamp: never project negative or >60% growth even in bull case
    base_growth = min(max(float(base_growth), 0.01), MAX_GROWTH_CAP)

    current_price = market_data.get("price")

    def _run(wacc_override, growth_override, tg, margin_mult):
        temp = {**wacc_data, "wacc": max(wacc_override, 0.01)}
        return run_dcf(
            financials, temp,
            manual_growth=growth_override,
            terminal_growth=tg,
            fcf_margin_multiplier=margin_mult,
        )

    # ── Bear ─────────────────────────────────────────────────────
    bear_wacc   = base_wacc + 0.02
    bear_growth = max(base_growth * 0.65, 0.03)    # growth meaningfully slower
    bear_tg     = 0.02
    bear_result = _run(bear_wacc, bear_growth, bear_tg, BEAR_FCF_MARGIN_COMPRESSION)

    # ── Base ─────────────────────────────────────────────────────
    base_result = _run(base_wacc, base_growth, 0.03, BASE_FCF_MARGIN_MULTIPLIER)

    # ── Bull ─────────────────────────────────────────────────────
    bull_wacc   = max(base_wacc - 0.02, 0.05)
    bull_growth = min(base_growth * 1.35, MAX_GROWTH_CAP)
    bull_tg     = 0.04
    bull_result = _run(bull_wacc, bull_growth, bull_tg, BULL_FCF_MARGIN_EXPANSION)

    def _upside(iv):
        if iv is not None and current_price and current_price > 0:
            return (iv - current_price) / current_price
        return None

    scenarios = {
        "Bear": {
            "wacc":             bear_wacc,
            "growth":           bear_growth,
            "terminal_growth":  bear_tg,
            "fcf_margin_mult":  BEAR_FCF_MARGIN_COMPRESSION,
            "description":      "+200 bps WACC | −35% growth | −20% margin",
            "intrinsic_value":  bear_result.get("intrinsic_value_per_share"),
            "upside_pct":       _upside(bear_result.get("intrinsic_value_per_share")),
        },
        "Base": {
            "wacc":             base_wacc,
            "growth":           base_growth,
            "terminal_growth":  0.03,
            "fcf_margin_mult":  BASE_FCF_MARGIN_MULTIPLIER,
            "description":      "Calculated WACC | TTM growth | Flat margin",
            "intrinsic_value":  base_result.get("intrinsic_value_per_share"),
            "upside_pct":       _upside(base_result.get("intrinsic_value_per_share")),
        },
        "Bull": {
            "wacc":             bull_wacc,
            "growth":           bull_growth,
            "terminal_growth":  bull_tg,
            "fcf_margin_mult":  BULL_FCF_MARGIN_EXPANSION,
            "description":      "−200 bps WACC | +35% growth | +20% margin",
            "intrinsic_value":  bull_result.get("intrinsic_value_per_share"),
            "upside_pct":       _upside(bull_result.get("intrinsic_value_per_share")),
        },
    }

    return scenarios
