"""
valuation.py — Main CLI entry point for the AI Sector Valuation Tool.

Usage
─────
  python valuation.py --ticker NVDA
  python valuation.py --ticker MSFT --peers GOOGL META AMD CRM --export
  python valuation.py --ticker AMD  --wacc 0.10 --growth 0.20

Flags
─────
  --ticker   (required) Target stock ticker
  --peers    Override default peer group
  --wacc     Manual WACC override (decimal, e.g. 0.10 = 10%)
  --growth   Manual Stage-1 revenue growth override (decimal, e.g. 0.25 = 25%)
  --export   Save charts to PNG files instead of displaying interactively
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tabulate import tabulate

from config  import DEFAULT_PEER_GROUP
from utils   import fetch_financials, build_wacc, get_market_data, format_large_number
from dcf     import dcf_summary
from comps   import comps_summary
from sensitivity import build_sensitivity_matrix, build_scenarios
from charts  import (
    chart_valuation_waterfall,
    chart_comps_ev_ebitda,
    chart_sensitivity_heatmap,
)


# ─── Header Banner ────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║        AI Sector Valuation Tool  ·  Bubble or Boom?             ║
║   DCF  ·  Comparable Company Analysis  ·  Sensitivity Analysis  ║
╚══════════════════════════════════════════════════════════════════╝"""


# ─── Print Helpers ────────────────────────────────────────────────────────────

def section(title: str, width: int = 66):
    """Print a styled section divider."""
    print(f"\n{'─' * width}")
    print(f"  ❯  {title}")
    print(f"{'─' * width}")


def fmt_pct(v, plus=True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{'+' if plus and v > 0 else ''}{v:.1%}"


def fmt_usd(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.2f}"


# ─── Section Printers ─────────────────────────────────────────────────────────

def print_dcf(dcf: dict, ticker: str):
    section(f"DCF Valuation  —  {ticker}")

    if "error" in dcf:
        print(f"  [Error] {dcf['error']}")
        return

    rows = [
        ["Intrinsic Value / Share",    fmt_usd(dcf.get("intrinsic_value_per_share"))],
        ["Current Market Price",       fmt_usd(dcf.get("current_price"))],
        ["Implied Upside / (Downside)",fmt_pct(dcf.get("upside_pct"))],
        ["Model Verdict",              dcf.get("verdict", "N/A")],
        ["─" * 30,                     "─" * 14],
        ["WACC Used",                  fmt_pct(dcf.get("wacc_used"), plus=False)],
        ["Stage-1 Revenue Growth",     fmt_pct(dcf.get("growth_rate_used"), plus=False)],
        ["Growth Source",              dcf.get("growth_source", "N/A")],
        ["Terminal Growth Rate",       fmt_pct(dcf.get("terminal_growth_used"), plus=False)],
        ["Terminal Value % of EV",     fmt_pct(dcf.get("tv_pct_of_ev"), plus=False)],
        ["FCF Margin Assumption",      dcf.get("fcf_margin_note", "N/A")],
        ["Implied Enterprise Value",   format_large_number(dcf.get("enterprise_value"))],
    ]
    print(tabulate(rows, tablefmt="plain", colalign=("left", "right")))


def print_comps(comps: dict, ticker: str):
    section(f"Comparable Company Analysis  —  {ticker}")

    df = comps.get("comps_df")
    if df is None or df.empty:
        print("  No comparable data available.")
        return

    # ── Multiples table ───────────────────────────────────────
    display = df[["ev_ebitda", "pe_ratio", "ps_ratio", "peg_ratio"]].copy()
    display.columns = ["EV/EBITDA", "P/E", "P/S", "PEG"]
    display = display.applymap(
        lambda x: f"{x:.1f}×" if pd.notna(x) and isinstance(x, (int, float)) and x > 0 else "—"
    )
    print()
    print(tabulate(display, headers="keys", tablefmt="rounded_outline"))

    # ── Sector medians ────────────────────────────────────────
    medians = comps.get("sector_medians", {})
    label_map = {"ev_ebitda": "EV/EBITDA", "pe_ratio": "P/E",
                 "ps_ratio": "P/S", "peg_ratio": "PEG"}
    med_rows = [
        [label_map[k], f"{v:.1f}×"]
        for k, v in medians.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if med_rows:
        print("\n  Sector Medians:")
        print(tabulate(med_rows, headers=["Multiple", "Median"], tablefmt="plain",
                       colalign=("left", "right")))

    # ── Implied prices ────────────────────────────────────────
    implied = comps.get("implied_prices", {})
    imp_rows = [
        [label_map.get(k, k), fmt_usd(v)]
        for k, v in implied.items() if v is not None and v > 0
    ]
    if imp_rows:
        print("\n  Implied Prices from Sector-Median Multiples:")
        print(tabulate(imp_rows, headers=["Method", "Implied Price"], tablefmt="plain",
                       colalign=("left", "right")))


def print_scenarios(scenarios: dict, ticker: str, current_price: float):
    section(f"Bear / Base / Bull Scenarios  —  {ticker}")

    rows = []
    for name, s in scenarios.items():
        rows.append([
            name,
            s.get("description", ""),
            fmt_pct(s.get("wacc"),            plus=False),
            fmt_pct(s.get("growth"),          plus=False),
            fmt_pct(s.get("terminal_growth"), plus=False),
            fmt_usd(s.get("intrinsic_value")),
            fmt_pct(s.get("upside_pct")),
        ])

    print()
    print(tabulate(
        rows,
        headers=["Scenario", "Assumptions", "WACC", "Growth", "Tgrowth",
                 "Intrinsic", "vs. Mkt"],
        tablefmt="rounded_outline",
        colalign=("left","left","right","right","right","right","right"),
    ))


def print_bubble_verdict(
    ticker: str,
    current_price: float,
    dcf: dict,
    comps: dict,
    scenarios: dict,
):
    """
    Aggregate all implied values, compare to market price, and render
    a plain-English bubble / fair-value verdict.
    """
    section(f"Bubble Verdict  —  {ticker}  @  {fmt_usd(current_price)}")

    all_implied: list[tuple[str, float]] = []

    # DCF base
    iv = dcf.get("intrinsic_value_per_share")
    if iv and not np.isnan(iv):
        all_implied.append(("DCF  (Base Case)", iv))

    # Comps
    label_map = {"ev_ebitda": "EV/EBITDA  Comps",
                 "pe_ratio":  "P/E  Comps",
                 "ps_ratio":  "P/S  Comps"}
    for k, v in comps.get("implied_prices", {}).items():
        if v and v > 0:
            all_implied.append((label_map.get(k, k), v))

    # Scenario DCFs
    for name, s in scenarios.items():
        iv_s = s.get("intrinsic_value")
        if iv_s and not np.isnan(iv_s):
            all_implied.append((f"DCF  {name}", iv_s))

    if not all_implied:
        print("  Insufficient data to render verdict.")
        return

    # ── Per-method table ──────────────────────────────────────
    upsides = []
    verdict_col = []
    rows = []
    for method, val in all_implied:
        upside = (val - current_price) / current_price
        upsides.append(upside)
        v_label = (
            "UNDERVALUED" if upside >  0.15 else
            "OVERVALUED"  if upside < -0.15 else
            "FAIR VALUE"
        )
        verdict_col.append(v_label)
        rows.append([method, fmt_usd(val), fmt_pct(upside), v_label])

    print()
    print(tabulate(rows,
                   headers=["Method", "Implied", "vs. Market", "Signal"],
                   tablefmt="rounded_outline",
                   colalign=("left", "right", "right", "left")))

    # ── Summary statistics ────────────────────────────────────
    avg_upside  = float(np.mean(upsides))
    n_over      = sum(1 for u in upsides if u < -0.15)
    n_under     = sum(1 for u in upsides if u >  0.15)
    n_total     = len(upsides)

    print(f"\n  Average Implied Upside / (Downside) :  {avg_upside:>+.1%}")
    print(f"  Models showing OVERVALUED           :  {n_over} / {n_total}")
    print(f"  Models showing UNDERVALUED          :  {n_under} / {n_total}")

    # ── Final verdict ─────────────────────────────────────────
    if avg_upside < -0.25 and n_over > n_total * 0.55:
        verdict = "LIKELY IN BUBBLE TERRITORY"
        color   = "⚠️ "
        detail  = (
            f"  {ticker} is priced well above fundamental estimates across\n"
            f"  {n_over}/{n_total} models. Reaching current valuations requires\n"
            f"  heroic assumptions about growth and margin expansion that have\n"
            f"  historically materialised for only a small fraction of companies.\n"
            f"  Risk/reward is heavily skewed to the downside."
        )
    elif avg_upside < -0.10:
        verdict = "MODERATELY OVERVALUED"
        color   = "⚡ "
        detail  = (
            f"  {ticker} trades at a premium to most fundamental models.\n"
            f"  A portion of the premium may be justified given AI tailwinds,\n"
            f"  but the margin of safety is thin. Outcomes depend heavily on\n"
            f"  AI monetisation and macro conditions."
        )
    elif avg_upside > 0.20 and n_under > n_total * 0.50:
        verdict = "POTENTIALLY UNDERVALUED"
        color   = "✅ "
        detail  = (
            f"  {ticker} appears below intrinsic value across {n_under}/{n_total} models.\n"
            f"  If growth assumptions prove conservative, meaningful upside remains.\n"
            f"  Validate with qualitative research on competitive positioning."
        )
    else:
        verdict = "FAIRLY VALUED  /  MIXED SIGNALS"
        color   = "⚖️  "
        detail  = (
            f"  {ticker} shows mixed signals. Current price broadly overlaps\n"
            f"  the range of fundamental estimates — outcome is highly sensitive\n"
            f"  to execution on AI product roadmaps, margin trajectory, and\n"
            f"  macro rate assumptions."
        )

    print(f"\n  ┌{'─' * 62}┐")
    print(f"  │  {color}VERDICT:  {verdict:<52}│")
    print(f"  └{'─' * 62}┘\n")
    print(detail)
    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  Disclaimer: point-in-time estimate from public data only.")
    print("  DCF outputs are highly sensitive to WACC and growth rate.")
    print("  Not investment advice.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="valuation.py",
        description="AI Sector Valuation Tool — DCF · Comps · Sensitivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python valuation.py --ticker NVDA\n"
            "  python valuation.py --ticker MSFT --peers GOOGL META AMD CRM --export\n"
            "  python valuation.py --ticker AMD  --wacc 0.10 --growth 0.20\n"
        ),
    )
    p.add_argument("--ticker", type=str, required=True,
                   help="Target stock ticker (e.g. NVDA)")
    p.add_argument("--peers", type=str, nargs="+", default=None,
                   help="Override default peer group (e.g. --peers MSFT GOOGL META)")
    p.add_argument("--wacc", type=float, default=None,
                   help="Manual WACC override as decimal (e.g. --wacc 0.10 → 10%%)")
    p.add_argument("--growth", type=float, default=None,
                   help="Manual Stage-1 revenue growth override (e.g. --growth 0.25 → 25%%)")
    p.add_argument("--export", action="store_true",
                   help="Save charts as PNG files instead of displaying interactively")
    return p


def main():
    args   = build_parser().parse_args()
    ticker = args.ticker.upper().strip()
    peers  = [p.upper() for p in args.peers] if args.peers else None

    print(BANNER)
    print(f"\n  Target   :  {ticker}")
    print(f"  Peers    :  {peers or DEFAULT_PEER_GROUP}")
    print(f"  WACC     :  {'manual ' + f'{args.wacc:.2%}' if args.wacc else 'calculated'}")
    print(f"  Growth   :  {'manual ' + f'{args.growth:.1%}' if args.growth else 'TTM / default'}")
    print(f"  Export   :  {'PNG files' if args.export else 'interactive charts'}")

    # ── 1. Fetch Data ─────────────────────────────────────────────────────────
    section("Fetching Data")
    print(f"  Pulling financials for {ticker} …")
    financials  = fetch_financials(ticker)
    market_data = get_market_data(ticker)
    wacc_data   = build_wacc(ticker, manual_wacc=args.wacc)

    current_price = market_data.get("price")
    if not current_price:
        print(f"\n  [Fatal] Cannot retrieve current price for {ticker}. "
              "Check the ticker and try again.")
        sys.exit(1)

    print(f"\n  Current Price     :  {fmt_usd(current_price)}")
    print(f"  Market Cap        :  {format_large_number(market_data.get('market_cap'))}")
    print(f"  Enterprise Value  :  {format_large_number(market_data.get('ev'))}")
    print(f"  Calculated WACC   :  {wacc_data['wacc']:.2%}"
          + (" (manual override)" if wacc_data.get("manual_override") else ""))
    print(f"  Beta              :  {wacc_data.get('beta', 'N/A')}")

    # ── 2. DCF ────────────────────────────────────────────────────────────────
    section("Running DCF Model")
    dcf = dcf_summary(financials, market_data, wacc_data, manual_growth=args.growth)
    print_dcf(dcf, ticker)

    # ── 3. Comps ──────────────────────────────────────────────────────────────
    comps = comps_summary(
        target_ticker=ticker,
        peers=peers,
        financials=financials,
        market_data=market_data,
    )
    print_comps(comps, ticker)

    # ── 4. Sensitivity ────────────────────────────────────────────────────────
    section(f"Sensitivity Analysis  —  {ticker}")
    print("  Building WACC × terminal-growth matrix …\n")
    sens_df = build_sensitivity_matrix(financials, market_data, wacc_data)
    print(tabulate(sens_df.round(2), headers="keys", tablefmt="rounded_outline",
                   floatfmt=",.2f"))

    scenarios = build_scenarios(financials, market_data, wacc_data,
                                manual_growth=args.growth)
    print_scenarios(scenarios, ticker, current_price)

    # ── 5. Charts ─────────────────────────────────────────────────────────────
    section("Generating Charts")

    dcf_iv = dcf.get("intrinsic_value_per_share") if "error" not in dcf else None

    print("  [1/3] Valuation Waterfall …")
    chart_valuation_waterfall(
        dcf_value      = dcf_iv,
        comps_implied  = comps.get("implied_prices", {}),
        current_price  = current_price,
        ticker         = ticker,
        export         = args.export,
        export_path    = f"{ticker}_waterfall.png",
    )

    print("  [2/3] Comps EV/EBITDA Bar Chart …")
    chart_comps_ev_ebitda(
        comps_df         = comps.get("comps_df", pd.DataFrame()),
        target_ticker    = ticker,
        median_ev_ebitda = comps.get("sector_medians", {}).get("ev_ebitda"),
        export           = args.export,
        export_path      = f"{ticker}_comps.png",
    )

    print("  [3/3] Sensitivity Heatmap …")
    chart_sensitivity_heatmap(
        sensitivity_df = sens_df,
        current_price  = current_price,
        ticker         = ticker,
        export         = args.export,
        export_path    = f"{ticker}_sensitivity.png",
    )

    # ── 6. Bubble Verdict ─────────────────────────────────────────────────────
    print_bubble_verdict(ticker, current_price, dcf, comps, scenarios)

    # ── Footer ────────────────────────────────────────────────────────────────
    print("═" * 66)
    print("  Analysis complete.")
    if args.export:
        print(f"  Charts saved to  {ticker}_waterfall.png  |  "
              f"{ticker}_comps.png  |  {ticker}_sensitivity.png")
    print("═" * 66)


if __name__ == "__main__":
    main()
