# AI Sector Valuation Tool — Methodology & Bubble Thesis

*A companion report for `valuation.py`*

---

## 1. Overview

This tool is a professional-grade financial modelling suite built entirely in Python. It performs three complementary analyses — Discounted Cash Flow (DCF), Comparable Company Analysis (Comps), and Sensitivity / Scenario Modelling — for any publicly traded stock, with a curated default peer group of leading AI-infrastructure and platform companies: NVIDIA, Microsoft, Alphabet, Meta, AMD, Amazon, and Salesforce.

The central research question guiding the tool's design is: **Are current AI-sector valuations justified by fundamentals, or are we witnessing a speculative bubble?** By anchoring market prices to independently derived intrinsic values across multiple methods, the tool provides a structured, evidence-based framework for evaluating that question — the kind of rigorous thinking expected of analysts at wealth management firms and equity research desks.

---

## 2. Methodology

### Discounted Cash Flow (DCF)

The DCF model projects ten years of free cash flow (FCF) and discounts them back to present value at the Weighted Average Cost of Capital (WACC). It uses a **revenue-based projection approach**: rather than growing FCF directly (which is unstable for companies with rapidly expanding margins), the model projects annual revenue using a two-stage growth schedule and applies a TTM FCF margin to each year's projected revenue.

**Stage 1 (Years 1–5):** Revenue grows at the Stage-1 rate, which defaults to the trailing twelve-month (TTM) year-over-year revenue growth, capped at 60% to prevent runaway projections.

**Stage 2 (Years 6–10):** Growth linearly decelerates from the Stage-1 rate to the terminal rate, mimicking the economic reality that even exceptional businesses eventually converge toward the pace of the broader economy.

**Terminal Value:** The Gordon Growth Model is applied to Year-10 FCF: TV = FCF₁₀ × (1 + g) ÷ (WACC − g), where g is the long-run nominal growth rate (default 3.0%, approximating long-run U.S. nominal GDP growth).

**WACC** is computed from first principles: CAPM is used for the cost of equity (risk-free rate + β × Equity Risk Premium), with the risk-free rate set to the 10-year Treasury yield (default 4.3%) and ERP of 5.5% (consistent with Damodaran's implied ERP estimates). The cost of debt is estimated from interest expense data where available.

Equity value = Enterprise Value − Total Debt + Cash & Equivalents, divided by diluted shares outstanding.

### Comparable Company Analysis (Comps)

The comps engine pulls live market data for the full AI-sector peer group and computes four trading multiples for each company:

- **EV/EBITDA** — the most capital-structure-neutral multiple; preferred when companies carry varying debt loads.
- **P/E (Trailing)** — straightforward earnings capitalisation; less meaningful for pre-profit or rapidly re-investing names.
- **P/S (Price-to-Sales)** — useful for early-stage or high-growth companies where EBITDA is depressed by heavy R&D spend.
- **PEG Ratio** — P/E adjusted for earnings growth rate; a higher PEG suggests the market is paying a greater premium for each unit of growth.

Sector **medians** (not means, to resist outlier distortion) are then applied to the target company's own EBITDA, revenue, and EPS to derive **implied share prices** — the standard "trading comps" output used in equity research pitchbooks.

### Sensitivity & Scenario Analysis

The sensitivity module sweeps WACC from the base estimate ±200 basis points (in 100-bps steps) and terminal growth from 1%–5% (in 1% steps), producing a 5×5 heatmap of intrinsic values. This answers the practitioner's most important question: *How wrong would my key assumptions need to be before the current price becomes defensible — or before the stock becomes dramatically cheap?*

Three **named scenarios** complement the grid:

| Scenario | WACC Adjustment | Growth | FCF Margin |
|----------|----------------|--------|------------|
| Bear | +200 bps | −35% vs. base | −20% compression |
| Base | Calculated | TTM historical | No change |
| Bull | −200 bps | +35% vs. base | +20% expansion |

---

## 3. The AI Bubble Question

The debate over AI valuations echoes — and departs from — the 1999–2001 dot-com episode in instructive ways.

**What would justify current prices?** For the highest-valued AI names, reaching consensus price targets through a DCF requires compounding assumptions: sustained 30–60% revenue growth for five or more years, continued margin expansion as compute costs fall and software attach rates rise, and a discount rate that stays structurally low. None of these are implausible individually — NVIDIA's data-centre revenue growth has in fact exceeded even bull-case assumptions for several consecutive quarters. Microsoft's Copilot and Azure AI services are gaining enterprise traction. The underlying technology (transformer-based LLMs, inference chips, CUDA-adjacent software stacks) is genuinely differentiated and exhibits strong switching costs and network effects — the characteristics that made past tech mega-caps durable compounders.

**What would signal a bubble?** The warning signs historically associated with asset bubbles are present in varying degrees: valuation multiples at or above dot-com peaks for certain names, heavy insider selling, venture and private-equity capital flooding into undifferentiated AI infrastructure plays, and narratives that treat total addressable market (TAM) projections as near-certainties. The critical stress test is the **terminal value dependency**: when a company's DCF is 85–95% terminal value, the model is telling you that the investment thesis rests almost entirely on what the business looks like a decade hence — a domain where forecast accuracy is low. A sensitivity run on NVIDIA or AMD at current prices typically shows that intrinsic value only exceeds market price in a narrow band of optimistic WACC and growth assumptions, which is a hallmark of a market pricing in the best-case scenario with limited margin of safety.

The dot-com parallel is instructive not because AI is necessarily a mirage — the underlying technology then (the internet) did transform the global economy — but because even transformative technologies can produce poor equity returns when purchased at prices that already discount perfection. Amazon, the canonical success story, fell 90% from its 1999 peak before eventually vindicating its long-term bull case. The lesson is not to avoid AI stocks but to price them with discipline.

---

## 4. Limitations

**Data quality and timeliness.** All data is sourced from Yahoo Finance via `yfinance`. Figures may lag earnings releases by days, and the income-statement parsing relies on field-label matching that can break when yfinance's schema changes.

**FCF volatility.** Free cash flow is inherently noisier than earnings — one-time working capital swings, lumpy capex cycles, or stock-based compensation accounting can distort the TTM margin used as the DCF's input. For companies still in heavy-investment phases (e.g. early cloud build-out), a revenue-based FCF projection with a normalised margin assumption may be more informative than mechanically using the trailing figure.

**No analyst-estimate integration.** The model falls back to TTM revenue growth where forward consensus estimates are unavailable. For companies with strong sell-side coverage, consensus revenue or EPS estimates for the next 1–2 years would improve Stage-1 accuracy. Adding a `--analyst-growth` flag backed by the `yfinance` earnings estimates table is a natural extension.

**Terminal value sensitivity.** For high-growth names, 70–90% of modelled value can sit in the terminal value. This is mathematically stable but should be treated as a caution flag, not a comfort. The sensitivity heatmap is precisely designed to surface this risk.

**Comparables consistency.** Peer groups are defined by the analyst. Including Meta alongside NVIDIA in a single peer group conflates an AI-infrastructure pure-play with an ad-tech platform that uses AI internally. For a formal equity research note, sub-peer-groups by business model would be more rigorous.

---

## 5. How to Use

### Installation

```bash
pip install yfinance pandas numpy matplotlib tabulate
```

### Basic Run

```bash
python valuation.py --ticker NVDA
```

### With Custom Peer Group

```bash
python valuation.py --ticker NVDA --peers MSFT GOOGL META AMD
```

### Override Assumptions

```bash
# Manual WACC of 10%, Stage-1 growth of 25%
python valuation.py --ticker AMD --wacc 0.10 --growth 0.25
```

### Export Charts to PNG

```bash
python valuation.py --ticker MSFT --export
# Produces: MSFT_waterfall.png, MSFT_comps.png, MSFT_sensitivity.png
```

### Interpreting the Output

1. **DCF section** — the intrinsic value per share and implied upside. Pay attention to the *growth source* and *terminal value %* lines; they tell you how dependent the estimate is on far-future assumptions.
2. **Comps table** — scan for outliers. A name trading at 2× the sector median EV/EBITDA is pricing in a fundamentally different growth or margin trajectory.
3. **Sensitivity heatmap** — find the WACC and terminal-growth coordinates that match the current price. If that point sits in the top-left corner (low WACC, high terminal growth), the market is priced for perfection.
4. **Bubble Verdict** — a consensus signal across all models. Treat it as a starting hypothesis, not a conclusion.

---

*Built with Python 3.10+ · yfinance · pandas · numpy · matplotlib · tabulate*
*Data sourced live from Yahoo Finance at runtime. Not investment advice.*
