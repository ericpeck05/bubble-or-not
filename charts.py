"""
charts.py — All visualizations for the AI Sector Valuation Tool.

Three charts:
  1. Valuation Waterfall — implied prices across methods vs. current price
  2. Comps Bar Chart     — EV/EBITDA multiples across the peer group
  3. Sensitivity Heatmap — WACC × terminal-growth intrinsic-value grid

All charts use a dark professional theme and include titles, axis labels,
and a data-source footer.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

from config import CHART_STYLE, CHART_FIGSIZE

# ─── Style Constants ──────────────────────────────────────────────────────────
FOOTER       = "Data: Yahoo Finance  |  Built with yfinance"
ACCENT       = "#00D4FF"   # electric blue — target ticker / current price
UP_GREEN     = "#27AE60"
DOWN_RED     = "#E74C3C"
BAR_BLUE     = "#2E86C1"
MEDIAN_AMBER = "#F39C12"

BG_DARK      = "#0F0F1A"
BG_PANEL     = "#1A1A2E"
SPINE_COLOR  = "#2D2D50"
TICK_COLOR   = "#AAAACC"
LABEL_COLOR  = "#CCCCDD"
TITLE_COLOR  = "#FFFFFF"


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def _style(fig, axes):
    """Apply unified dark styling to a figure and its axes."""
    fig.patch.set_facecolor(BG_DARK)
    for ax in (np.array(axes).flatten() if not isinstance(axes, plt.Axes) else [axes]):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TICK_COLOR, labelsize=9)
        ax.xaxis.label.set_color(LABEL_COLOR)
        ax.yaxis.label.set_color(LABEL_COLOR)
        ax.title.set_color(TITLE_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
            spine.set_linewidth(0.8)
    return fig


def _footer(fig):
    fig.text(0.99, 0.01, FOOTER, ha="right", va="bottom",
             fontsize=7, color="#555577", style="italic")


def _save_or_show(fig, export: bool, path: str):
    if export:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        try:
            plt.show()
        except Exception as exc:
            # No interactive display — save a fallback PNG and inform the user
            fallback = path.replace(".png", "_auto.png")
            fig.savefig(fallback, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  [Info] No interactive display available. Chart saved to {fallback}")
    plt.close(fig)


# ─── Chart 1: Valuation Waterfall ────────────────────────────────────────────

def chart_valuation_waterfall(
    dcf_value: float,
    comps_implied: dict,
    current_price: float,
    ticker: str,
    export: bool = False,
    export_path: str = None,
):
    """
    Horizontal bar chart: one bar per valuation method, sorted by implied value.
    A vertical dashed line marks the live market price.
    Bars are green when implied > current and red when implied < current.
    Right-side annotations show % upside / downside.
    """
    method_labels = {
        "dcf":       "DCF  (Intrinsic)",
        "ev_ebitda": "EV/EBITDA  Comps",
        "pe_ratio":  "P/E  Comps",
        "ps_ratio":  "P/S  Comps",
    }

    methods, values = [], []
    if dcf_value is not None:
        methods.append(method_labels["dcf"])
        values.append(dcf_value)
    for key in ("ev_ebitda", "pe_ratio", "ps_ratio"):
        val = comps_implied.get(key)
        if val is not None and not np.isnan(val) and val > 0:
            methods.append(method_labels[key])
            values.append(val)

    if not values:
        print("  [Warning] No valuation data — skipping waterfall chart.")
        return

    # Sort ascending so largest bar is at the top
    pairs   = sorted(zip(values, methods))
    values  = [p[0] for p in pairs]
    methods = [p[1] for p in pairs]
    colors  = [UP_GREEN if v >= current_price else DOWN_RED for v in values]

    with plt.style.context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

        bars = ax.barh(methods, values, color=colors, height=0.50, alpha=0.88,
                       edgecolor=SPINE_COLOR, linewidth=0.6)

        # Current price line
        ax.axvline(x=current_price, color=ACCENT, linestyle="--", linewidth=2.0,
                   label=f"Market Price  ${current_price:.2f}", zorder=5)

        # Value labels inside / beside bars
        x_max = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(
                val + x_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.2f}",
                va="center", ha="left",
                color="#EEEEEE", fontsize=10, fontweight="bold",
            )

        # % annotations on the far right
        right_edge = ax.get_xlim()[1] if ax.get_xlim()[1] > x_max else x_max * 1.25
        for bar, val in zip(bars, values):
            pct  = (val - current_price) / current_price * 100
            sign = "+" if pct >= 0 else ""
            ax.text(
                x_max * 1.22,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{pct:.1f}%",
                va="center", ha="left",
                color=UP_GREEN if pct >= 0 else DOWN_RED,
                fontsize=9, fontweight="bold",
            )

        ax.set_xlabel("Implied Share Price (USD)", fontsize=11, labelpad=8)
        ax.set_title(
            f"{ticker}  —  Cross-Model Valuation Summary",
            fontsize=14, fontweight="bold", pad=16, color=TITLE_COLOR,
        )
        ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=SPINE_COLOR,
                  labelcolor=LABEL_COLOR, loc="lower right")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

        _style(fig, ax)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.03, 0.90, 1])

    _save_or_show(fig, export, export_path or f"{ticker}_waterfall.png")


# ─── Chart 2: Comps EV/EBITDA Bar Chart ──────────────────────────────────────

def chart_comps_ev_ebitda(
    comps_df: pd.DataFrame,
    target_ticker: str,
    median_ev_ebitda: float,
    export: bool = False,
    export_path: str = None,
):
    """
    Vertical bar chart of EV/EBITDA multiples across all peers.
    Target ticker bar is highlighted in accent blue.
    Sector median shown as a dashed amber line.
    """
    if comps_df is None or comps_df.empty:
        print("  [Warning] No comps data — skipping EV/EBITDA chart.")
        return

    df = (comps_df[["ev_ebitda"]]
          .dropna()
          .loc[lambda x: x["ev_ebitda"] > 0]
          .sort_values("ev_ebitda", ascending=False))

    if df.empty:
        print("  [Warning] All EV/EBITDA values are N/A — skipping chart.")
        return

    tickers = df.index.tolist()
    vals    = df["ev_ebitda"].tolist()
    colors  = [ACCENT if t == target_ticker else BAR_BLUE for t in tickers]

    with plt.style.context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

        bars = ax.bar(tickers, vals, color=colors, alpha=0.88,
                      edgecolor=SPINE_COLOR, linewidth=0.6)

        # Median line
        if median_ev_ebitda and not np.isnan(median_ev_ebitda):
            ax.axhline(
                y=median_ev_ebitda, color=MEDIAN_AMBER, linestyle="--", linewidth=2.0,
                label=f"Sector Median  {median_ev_ebitda:.1f}×", zorder=4,
            )

        # Value labels above bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.012,
                f"{val:.1f}×",
                ha="center", va="bottom", color="#EEEEEE", fontsize=9,
            )

        # Highlight target label
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == target_ticker:
                lbl.set_color(ACCENT)
                lbl.set_fontweight("bold")

        ax.set_xlabel("Company", fontsize=11, labelpad=8)
        ax.set_ylabel("EV / EBITDA  (×)", fontsize=11, labelpad=8)
        ax.set_title(
            "EV/EBITDA  —  Comparable Analysis",
            fontsize=14, fontweight="bold", pad=16, color=TITLE_COLOR,
        )
        ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=SPINE_COLOR,
                  labelcolor=LABEL_COLOR)
        ax.set_ylim(bottom=0, top=max(vals) * 1.18)

        _style(fig, ax)
        _footer(fig)
        plt.tight_layout()

    _save_or_show(fig, export, export_path or f"{target_ticker}_comps.png")


# ─── Chart 3: Sensitivity Heatmap ────────────────────────────────────────────

def chart_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    current_price: float,
    ticker: str,
    export: bool = False,
    export_path: str = None,
):
    """
    Color-coded heatmap of intrinsic values over the WACC × terminal-growth grid.

    Color scale diverges at the current market price:
        Green tones  → implied value ABOVE current price (undervalued)
        Red tones    → implied value BELOW current price (overvalued)
    Each cell is annotated with the dollar value.
    """
    if sensitivity_df is None or sensitivity_df.empty:
        print("  [Warning] No sensitivity data — skipping heatmap.")
        return

    data = sensitivity_df.values.astype(float)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Diverging norm centred on current price
    vcenter = np.clip(current_price, vmin + 1, vmax - 1)
    try:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    except Exception:
        norm = None

    cmap = plt.cm.RdYlGn   # Red → Yellow → Green

    with plt.style.context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")

        # Axis labels
        ax.set_xticks(range(len(sensitivity_df.columns)))
        ax.set_xticklabels(sensitivity_df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(sensitivity_df.index)))
        ax.set_yticklabels(sensitivity_df.index, fontsize=9)

        # Cell annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    # Choose black or white text based on cell brightness
                    normed = (val - vmin) / max(vmax - vmin, 1)
                    txt_color = "black" if 0.30 < normed < 0.72 else "white"
                    ax.text(j, i, f"${val:,.0f}",
                            ha="center", va="center",
                            fontsize=9, fontweight="bold", color=txt_color)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.04)
        cbar.set_label("Intrinsic Value / Share (USD)", color=LABEL_COLOR, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=TICK_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TICK_COLOR)

        # Current-price indicator on colorbar
        if vmin < current_price < vmax:
            cbar_pos = (current_price - vmin) / max(vmax - vmin, 1)
            cbar.ax.axhline(y=cbar_pos, color=ACCENT, linewidth=2.0,
                            label=f"Market ${current_price:.0f}")

        ax.set_xlabel("Terminal Growth Rate", fontsize=11, labelpad=10)
        ax.set_ylabel("WACC", fontsize=11, labelpad=10)
        ax.set_title(
            f"{ticker}  —  DCF Sensitivity  |  WACC  ×  Terminal Growth Rate\n"
            f"Green = above market price ${current_price:.2f}   |   Red = below",
            fontsize=12, fontweight="bold", pad=16, color=TITLE_COLOR,
        )

        _style(fig, ax)
        _footer(fig)
        plt.tight_layout()

    _save_or_show(fig, export, export_path or f"{ticker}_sensitivity.png")
