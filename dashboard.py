"""
dashboard.py — Streamlit interactive dashboard for the AI Sector Valuation Tool.

Run with:
    streamlit run dashboard.py

All analysis engines (dcf.py, comps.py, sensitivity.py) are used as-is.
Charts are rebuilt here using Plotly for interactive hover / zoom.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Page config — MUST be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="AI Valuation · Bubble or Boom?",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Engine imports ─────────────────────────────────────────────────────────────
from config      import DEFAULT_PEER_GROUP
from utils       import fetch_financials, build_wacc, get_market_data, format_large_number
from dcf         import dcf_summary
from comps       import comps_summary
from sensitivity import build_sensitivity_matrix, build_scenarios

# ── Palette & constants ───────────────────────────────────────────────────────
ACCENT = "#00D4FF"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
AMBER  = "#F39C12"
BLUE   = "#2E86C1"
PT     = "plotly_dark"      # Plotly template

ALL_TICKERS = sorted(set(
    DEFAULT_PEER_GROUP + ["TSLA", "AAPL", "INTC", "QCOM", "SNOW", "PLTR", "ARM"]
))


# ── Cached data fetchers ───────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_ticker(ticker: str):
    return fetch_financials(ticker), get_market_data(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_wacc(ticker: str, manual_wacc):
    return build_wacc(ticker, manual_wacc=manual_wacc)

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_comps(target: str, peers_tuple: tuple):
    """Cache keyed on (target, peers). Market data fetched internally."""
    md = get_market_data(target)
    return comps_summary(target_ticker=target, peers=list(peers_tuple), market_data=md)


# ── Plotly chart builders ──────────────────────────────────────────────────────

def fig_waterfall(dcf_iv, comps_implied, current_price, ticker):
    """Horizontal bar chart: one bar per model, vertical line = market price."""
    label_map = {
        "ev_ebitda": "EV/EBITDA  Comps",
        "pe_ratio":  "P/E  Comps",
        "ps_ratio":  "P/S  Comps",
    }
    methods, values = [], []
    if dcf_iv:
        methods.append("DCF  (Intrinsic)")
        values.append(dcf_iv)
    for k, lbl in label_map.items():
        v = comps_implied.get(k)
        if v and v > 0 and not np.isnan(v):
            methods.append(lbl)
            values.append(v)
    if not values:
        return None

    pairs   = sorted(zip(values, methods))
    values  = [p[0] for p in pairs]
    methods = [p[1] for p in pairs]
    pcts    = [(v - current_price) / current_price * 100 for v in values]
    colors  = [GREEN if v >= current_price else RED for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=methods, orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"${v:,.2f}  ({'+' if p >= 0 else ''}{p:.1f}%)" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="%{y}<br>Implied: $%{x:,.2f}<extra></extra>",
    ))
    fig.add_vline(
        x=current_price, line_dash="dash", line_color=ACCENT, line_width=2,
        annotation_text=f"  Market  ${current_price:.2f}",
        annotation_font_color=ACCENT, annotation_font_size=11,
    )
    fig.update_layout(
        template=PT, title=f"{ticker} — Valuation Across Models",
        xaxis_title="Implied Share Price (USD)", xaxis_tickprefix="$",
        showlegend=False, margin=dict(l=10, r=170, t=50, b=30), height=300,
    )
    return fig


def fig_ev_ebitda(comps_df, target_ticker, median):
    """Bar chart of EV/EBITDA multiples — target highlighted in accent blue."""
    df = comps_df[["ev_ebitda"]].dropna()
    df = df[df["ev_ebitda"] > 0].sort_values("ev_ebitda", ascending=False)
    if df.empty:
        return None

    tickers = df.index.tolist()
    vals    = df["ev_ebitda"].tolist()
    colors  = [ACCENT if t == target_ticker else BLUE for t in tickers]

    fig = go.Figure(go.Bar(
        x=tickers, y=vals, marker_color=colors, opacity=0.85,
        text=[f"{v:.1f}×" for v in vals], textposition="outside",
        hovertemplate="%{x}  EV/EBITDA: %{y:.1f}×<extra></extra>",
    ))
    if median and not np.isnan(median):
        fig.add_hline(
            y=median, line_dash="dash", line_color=AMBER, line_width=2,
            annotation_text=f"  Median  {median:.1f}×",
            annotation_font_color=AMBER, annotation_font_size=11,
        )
    fig.update_layout(
        template=PT, title="AI Sector — EV/EBITDA Multiples",
        yaxis_title="EV / EBITDA  (×)", showlegend=False,
        margin=dict(l=10, r=20, t=50, b=30), height=340,
    )
    return fig


def fig_heatmap(sens_df, current_price, ticker):
    """Color-coded sensitivity heatmap with interactive hover. Green = above market."""
    data   = sens_df.values.astype(float)
    wacc_l = sens_df.index.tolist()
    tg_l   = sens_df.columns.tolist()

    hover = [[
        f"WACC: {wacc_l[i]}<br>Terminal Growth: {tg_l[j]}<br>"
        f"Intrinsic Value: ${data[i, j]:,.2f}<br>"
        f"vs. Market: {(data[i, j] - current_price) / current_price:+.1%}"
        for j in range(len(tg_l))]
        for i in range(len(wacc_l))]

    fig = go.Figure(go.Heatmap(
        z=data, x=tg_l, y=wacc_l,
        colorscale="RdYlGn", zmid=current_price,
        text=[[f"${v:,.0f}" for v in row] for row in data],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertext=hover, hoverinfo="text",
        colorbar=dict(title=dict(text="Intrinsic Value"), tickprefix="$"),
    ))
    fig.update_layout(
        template=PT,
        title=f"{ticker} — DCF Sensitivity: WACC × Terminal Growth  "
              f"(green = above market ${current_price:.0f})",
        xaxis_title="Terminal Growth Rate", yaxis_title="WACC",
        margin=dict(l=10, r=20, t=55, b=30), height=370,
    )
    return fig


def fig_fcf(dcf_result):
    """Grouped bar: projected FCF vs. discounted PV by year."""
    projected = dcf_result.get("projected_fcfs", [])
    pv_fcfs   = dcf_result.get("pv_fcfs", [])
    if not projected:
        return None

    years = [f"Yr {i + 1}" for i in range(len(projected))]
    proj  = [v / 1e9 for v in projected]
    pv    = [v / 1e9 for v in pv_fcfs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Projected FCF", x=years, y=proj,
        marker_color=BLUE, opacity=0.80,
        hovertemplate="Year %{x}<br>FCF: $%{y:.2f}B<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="PV of FCF (discounted)", x=years, y=pv,
        marker_color=ACCENT, opacity=0.80,
        hovertemplate="Year %{x}<br>PV: $%{y:.2f}B<extra></extra>",
    ))
    fig.update_layout(
        template=PT, title="10-Year Free Cash Flow Projection",
        yaxis_title="$ Billions", barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=55, b=30), height=320,
    )
    return fig


def fig_scenarios(scenarios, current_price):
    """Bar chart: Bear / Base / Bull intrinsic values vs. market price."""
    names  = list(scenarios.keys())
    values = [s.get("intrinsic_value") or 0 for s in scenarios.values()]
    pcts   = [(v - current_price) / current_price * 100 for v in values]
    colors = [RED, BLUE, GREEN]

    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors, opacity=0.85,
        text=[f"${v:,.2f}<br>{'+' if p >= 0 else ''}{p:.1f}%" for v, p in zip(values, pcts)],
        textposition="outside",
        hovertemplate="%{x}<br>Intrinsic: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(
        y=current_price, line_dash="dash", line_color=ACCENT, line_width=2,
        annotation_text=f"  Market  ${current_price:.2f}",
        annotation_font_color=ACCENT, annotation_font_size=11,
    )
    fig.update_layout(
        template=PT, title="Bear / Base / Bull Scenario Intrinsic Values",
        yaxis_title="Intrinsic Value / Share (USD)", yaxis_tickprefix="$",
        showlegend=False, margin=dict(l=10, r=10, t=55, b=30), height=340,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AI Valuation Tool")
    st.markdown("*Bubble or Boom?*")
    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol", value="NVDA", max_chars=10,
        help="Any US-listed equity. Optimised for AI / tech sector names.",
    ).upper().strip()

    peers_input = st.multiselect(
        "Peer Group",
        options=ALL_TICKERS,
        default=[t for t in DEFAULT_PEER_GROUP if t != "NVDA"],
        help="Tickers used for comparable company analysis.",
    )

    st.divider()
    st.markdown("**Override Assumptions**")

    toggle_col1, toggle_col2 = st.columns(2)
    use_wacc   = toggle_col1.toggle("WACC",   value=False)
    use_growth = toggle_col2.toggle("Growth", value=False)

    manual_wacc   = None
    manual_growth = None

    if use_wacc:
        manual_wacc = st.slider(
            "WACC (%)", min_value=5.0, max_value=25.0, value=10.0, step=0.5,
            format="%.1f%%",
        ) / 100

    if use_growth:
        manual_growth = st.slider(
            "Stage-1 Growth (%)", min_value=1.0, max_value=60.0, value=25.0, step=1.0,
            format="%.0f%%",
        ) / 100

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    if "last_run" in st.session_state:
        st.caption(f"Last run: **{st.session_state['last_run']}**")

    st.divider()
    st.caption("Data: Yahoo Finance via yfinance · Not investment advice")


# ── Analysis runner ────────────────────────────────────────────────────────────
if run_btn:
    if not ticker_input:
        st.error("Please enter a ticker symbol.")
        st.stop()

    peers_clean = [p for p in peers_input if p != ticker_input]

    with st.status(f"Analyzing **{ticker_input}** …", expanded=True) as status:

        st.write(f"📡 Fetching financials for **{ticker_input}** …")
        financials, market_data = _fetch_ticker(ticker_input)

        if not market_data.get("price"):
            status.update(label="Failed", state="error")
            st.error(f"Could not retrieve price data for **{ticker_input}**. "
                     "Check the ticker symbol and try again.")
            st.stop()

        st.write("⚙️ Building WACC …")
        wacc_data = _fetch_wacc(ticker_input, manual_wacc)

        st.write("📐 Running DCF model …")
        dcf = dcf_summary(financials, market_data, wacc_data, manual_growth=manual_growth)

        st.write(f"🔍 Fetching peer data ({len(peers_clean)} companies) …")
        comps = _fetch_comps(ticker_input, tuple(peers_clean))

        st.write("🌡️ Building sensitivity matrix …")
        sens_df   = build_sensitivity_matrix(financials, market_data, wacc_data)
        scenarios = build_scenarios(
            financials, market_data, wacc_data, manual_growth=manual_growth
        )

        status.update(label="Analysis complete ✓", state="complete", expanded=False)

    st.session_state["results"] = {
        "ticker":      ticker_input,
        "financials":  financials,
        "market_data": market_data,
        "wacc_data":   wacc_data,
        "dcf":         dcf,
        "comps":       comps,
        "sens_df":     sens_df,
        "scenarios":   scenarios,
    }
    st.session_state["last_run"] = ticker_input


# ── Landing page ───────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.markdown("""
    # 📊 AI Sector Valuation Tool
    ### *Bubble or Boom?*

    A professional equity analysis suite combining **DCF modelling**,
    **comparable company analysis**, and **sensitivity analysis**
    for AI-sector stocks.

    ---
    Enter a ticker in the sidebar and click **▶ Run Analysis** to begin.

    **Default peer group:** NVDA · MSFT · GOOGL · META · AMD · AMZN · CRM
    """)
    st.stop()


# ── Unpack session state ───────────────────────────────────────────────────────
R             = st.session_state["results"]
ticker        = R["ticker"]
market_data   = R["market_data"]
wacc_data     = R["wacc_data"]
dcf           = R["dcf"]
comps         = R["comps"]
sens_df       = R["sens_df"]
scenarios     = R["scenarios"]
financials    = R["financials"]

current_price = market_data.get("price", 0)
dcf_iv        = dcf.get("intrinsic_value_per_share") if "error" not in dcf else None
dcf_upside    = dcf.get("upside_pct")
verdict       = dcf.get("verdict", "N/A")


# ── Header ─────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([4, 1])
with hdr_left:
    st.markdown(f"# {ticker} &nbsp;&nbsp; `${current_price:.2f}`")
with hdr_right:
    if verdict == "UNDERVALUED":
        st.success(f"✅ {verdict}")
    elif verdict == "OVERVALUED":
        st.error(f"⚠️ {verdict}")
    else:
        st.info(f"⚖️ {verdict}")

# Metric strip
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Price",      f"${current_price:.2f}")
m2.metric("Market Cap",         format_large_number(market_data.get("market_cap")))
m3.metric("Enterprise Value",   format_large_number(market_data.get("ev")))
m4.metric("WACC",               f"{wacc_data['wacc']:.2%}")
m5.metric(
    "DCF Intrinsic Value",
    f"${dcf_iv:.2f}" if dcf_iv else "N/A",
    delta=f"{dcf_upside:+.1%}" if dcf_upside is not None else None,
)

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_over, tab_dcf, tab_comps, tab_sens = st.tabs(
    ["📋  Overview", "📐  DCF", "🔍  Comps", "🌡️  Sensitivity"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_over:
    col_stats, col_chart = st.columns([1, 1.7])

    with col_stats:
        st.markdown("#### Key Financials")
        fin = financials
        rows = {
            "Revenue (TTM)":    format_large_number(fin.get("revenue")),
            "EBITDA (TTM)":     format_large_number(fin.get("ebitda")),
            "FCF (TTM)":        format_large_number(fin.get("fcf")),
            "FCF Margin":       f"{fin['fcf_margin']:.1%}" if fin.get("fcf_margin") else "N/A",
            "Revenue Growth":   f"{fin['revenue_growth']:.1%}" if fin.get("revenue_growth") else "N/A",
            "Beta":             str(wacc_data.get("beta", "N/A")),
            "Cost of Equity":   f"{wacc_data['cost_of_equity']:.2%}" if wacc_data.get("cost_of_equity") else "N/A",
            "WACC":             f"{wacc_data['wacc']:.2%}",
        }
        st.dataframe(
            pd.DataFrame(rows.items(), columns=["Metric", "Value"]).set_index("Metric"),
            use_container_width=True,
        )

    with col_chart:
        wf = fig_waterfall(dcf_iv, comps.get("implied_prices", {}), current_price, ticker)
        if wf:
            st.plotly_chart(wf, use_container_width=True)
        else:
            st.info("Not enough valuation data to render waterfall chart.")

    # ── Bubble Verdict ─────────────────────────────────────────────────────
    st.markdown("#### Bubble Verdict")

    all_implied: list = []
    if dcf_iv:
        all_implied.append(("DCF  Base", dcf_iv))
    lmap = {"ev_ebitda": "EV/EBITDA", "pe_ratio": "P/E", "ps_ratio": "P/S"}
    for k, v in comps.get("implied_prices", {}).items():
        if v and v > 0:
            all_implied.append((lmap.get(k, k), v))
    for name, s in scenarios.items():
        iv = s.get("intrinsic_value")
        if iv:
            all_implied.append((f"DCF  {name}", iv))

    if all_implied:
        upsides = [(v - current_price) / current_price for _, v in all_implied]
        avg_up  = float(np.mean(upsides))
        n_over  = sum(1 for u in upsides if u < -0.15)
        n_under = sum(1 for u in upsides if u >  0.15)
        n_total = len(upsides)

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Avg. Implied Upside",  f"{avg_up:+.1%}")
        vc2.metric("Models: Overvalued",   f"{n_over} / {n_total}")
        vc3.metric("Models: Undervalued",  f"{n_under} / {n_total}")

        if avg_up < -0.20 and n_over > n_total * 0.55:
            st.error(
                "⚠️ **LIKELY IN BUBBLE TERRITORY** — Current price materially exceeds "
                "fundamental estimates across the majority of models."
            )
        elif avg_up < -0.10:
            st.warning(
                "⚡ **MODERATELY OVERVALUED** — Trades at a premium to most models. "
                "Margin of safety is thin."
            )
        elif avg_up > 0.15 and n_under > n_total * 0.50:
            st.success(
                "✅ **POTENTIALLY UNDERVALUED** — Implied upside across the majority of models."
            )
        else:
            st.info(
                "⚖️ **FAIRLY VALUED / MIXED SIGNALS** — Current price broadly overlaps "
                "the range of fundamental estimates."
            )

        verdict_rows = [
            {
                "Method":        m,
                "Implied Price": f"${v:,.2f}",
                "vs. Market":    f"{(v - current_price) / current_price:+.1%}",
                "Signal": (
                    "✅ UNDERVALUED" if (v - current_price) / current_price >  0.15 else
                    "⚠️ OVERVALUED"  if (v - current_price) / current_price < -0.15 else
                    "⚖️ FAIR VALUE"
                ),
            }
            for m, v in all_implied
        ]
        st.dataframe(pd.DataFrame(verdict_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DCF
# ══════════════════════════════════════════════════════════════════════════════
with tab_dcf:
    if "error" in dcf:
        st.error(f"DCF Error: {dcf['error']}")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Intrinsic Value",   f"${dcf_iv:.2f}" if dcf_iv else "N/A",
            delta=f"{dcf_upside:+.1%}" if dcf_upside is not None else None,
        )
        d2.metric("WACC Used",       f"{dcf.get('wacc_used', 0):.2%}")
        d3.metric("Stage-1 Growth",  f"{dcf.get('growth_rate_used', 0):.1%}")
        d4.metric("Terminal Value %", f"{dcf.get('tv_pct_of_ev', 0):.1%}" if dcf.get("tv_pct_of_ev") else "N/A")

        st.divider()

        col_fcf, col_detail = st.columns([1.6, 1])

        with col_fcf:
            fcf_fig = fig_fcf(dcf)
            if fcf_fig:
                st.plotly_chart(fcf_fig, use_container_width=True)

        with col_detail:
            st.markdown("#### Model Inputs & Outputs")
            dcf_table = {
                "Intrinsic Value / Share":  f"${dcf_iv:.2f}" if dcf_iv else "N/A",
                "Current Price":            f"${current_price:.2f}",
                "Implied Upside":           f"{dcf_upside:+.1%}" if dcf_upside else "N/A",
                "Enterprise Value":         format_large_number(dcf.get("enterprise_value")),
                "Equity Value":             format_large_number(dcf.get("equity_value")),
                "WACC":                     f"{dcf.get('wacc_used', 0):.2%}",
                "Stage-1 Growth":           f"{dcf.get('growth_rate_used', 0):.1%}",
                "Growth Source":            dcf.get("growth_source", "N/A"),
                "Terminal Growth":          f"{dcf.get('terminal_growth_used', 0):.1%}",
                "Terminal Value %":         f"{dcf.get('tv_pct_of_ev', 0):.1%}" if dcf.get("tv_pct_of_ev") else "N/A",
                "FCF Margin":               dcf.get("fcf_margin_note", "N/A"),
            }
            st.dataframe(
                pd.DataFrame(dcf_table.items(), columns=["Field", "Value"]).set_index("Field"),
                use_container_width=True,
            )

        with st.expander("📌 Model Methodology"):
            g1 = dcf.get("growth_rate_used", 0)
            g2 = dcf.get("terminal_growth_used", 0)
            st.markdown(f"""
**Two-stage revenue growth model:**

- **Stage 1 (Years 1–5):** `{g1:.1%}` annually *(source: {dcf.get('growth_source', 'N/A')})*
- **Stage 2 (Years 6–10):** Linear deceleration `{g1:.1%}` → `{g2:.1%}` (terminal rate)
- **Terminal Value:** Gordon Growth Model — `TV = FCF₁₀ × (1 + g) ÷ (WACC − g)`
- **FCF Projection:** Revenue-based — projected revenue × TTM FCF margin *(more stable than growing FCF directly)*
- Terminal value represents **{dcf.get('tv_pct_of_ev', 0):.1%}** of total enterprise value
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Comps
# ══════════════════════════════════════════════════════════════════════════════
with tab_comps:
    comps_df = comps.get("comps_df")
    medians  = comps.get("sector_medians", {})
    implied  = comps.get("implied_prices", {})

    if comps_df is None or comps_df.empty:
        st.warning("No comparable company data available.")
    else:
        ev_fig = fig_ev_ebitda(comps_df, ticker, medians.get("ev_ebitda"))
        if ev_fig:
            st.plotly_chart(ev_fig, use_container_width=True)

        col_tbl, col_imp = st.columns([2, 1])

        with col_tbl:
            st.markdown("#### Trading Multiples")
            display = comps_df[["ev_ebitda", "pe_ratio", "ps_ratio", "peg_ratio"]].copy()
            display.columns = ["EV/EBITDA", "P/E", "P/S", "PEG"]
            display = display.map(
                lambda x: f"{x:.1f}×" if pd.notna(x) and isinstance(x, (int, float)) and x > 0 else "—"
            )
            st.dataframe(display, use_container_width=True)

        with col_imp:
            st.markdown("#### Sector Medians")
            med_rows = [
                {"Multiple": lmap.get(k, k), "Median": f"{v:.1f}×"}
                for k, v in medians.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            if med_rows:
                st.dataframe(pd.DataFrame(med_rows), use_container_width=True, hide_index=True)

            st.markdown("#### Implied Prices")
            imp_rows = [
                {
                    "Method":  lmap.get(k, k),
                    "Implied": f"${v:.2f}",
                    "vs. Mkt": f"{(v - current_price) / current_price:+.1%}",
                }
                for k, v in implied.items() if v and v > 0
            ]
            if imp_rows:
                st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, hide_index=True)
            elif not imp_rows:
                st.caption("Insufficient data to compute implied prices.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
with tab_sens:
    hm = fig_heatmap(sens_df, current_price, ticker)
    if hm:
        st.plotly_chart(hm, use_container_width=True)

    st.divider()

    col_sc_tbl, col_sc_chart = st.columns([1, 1.5])

    with col_sc_tbl:
        st.markdown("#### Scenario Analysis")
        sc_rows = [
            {
                "Scenario":    name,
                "Assumptions": s.get("description", ""),
                "WACC":        f"{s.get('wacc', 0):.2%}",
                "Growth":      f"{s.get('growth', 0):.1%}",
                "Value":       f"${s.get('intrinsic_value', 0):,.2f}" if s.get("intrinsic_value") else "N/A",
                "vs. Mkt":     f"{s.get('upside_pct', 0):+.1%}" if s.get("upside_pct") is not None else "N/A",
            }
            for name, s in scenarios.items()
        ]
        st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    with col_sc_chart:
        sc_fig = fig_scenarios(scenarios, current_price)
        if sc_fig:
            st.plotly_chart(sc_fig, use_container_width=True)
