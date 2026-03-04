"""
config.py — Constants, peer group definitions, and sector defaults.
All tunable parameters live here to make the model easy to adjust.
"""

# ─── Peer Group ───────────────────────────────────────────────────────────────
# Default comparable peer group — change freely or override via the dashboard.
DEFAULT_PEER_GROUP = ["NVDA", "MSFT", "GOOGL", "META", "AMD", "AMZN", "CRM"]

# ─── Discount Rate Assumptions ────────────────────────────────────────────────
# 10-year U.S. Treasury yield (used as risk-free rate). Updated periodically.
RISK_FREE_RATE: float = 0.043          # 4.3% — default; fetched live if possible
EQUITY_RISK_PREMIUM: float = 0.055    # 5.5% — Damodaran implied ERP (historical avg)
TERMINAL_GROWTH_RATE: float = 0.030   # 3.0% — roughly in line with long-run nominal GDP

# ─── DCF Model Horizon ────────────────────────────────────────────────────────
PROJECTION_YEARS: int = 10            # Two 5-year stages

# ─── Sensitivity Grid ─────────────────────────────────────────────────────────
# WACC range: base ± 200 bps in 100-bps steps (5 points)
WACC_BPS_STEPS: list = [-200, -100, 0, 100, 200]

# Terminal growth grid: 1% → 5% in 1% steps
TERMINAL_GROWTH_RANGE: list = [0.01, 0.02, 0.03, 0.04, 0.05]

# ─── Scenario FCF Margin Multipliers ──────────────────────────────────────────
BEAR_FCF_MARGIN_COMPRESSION: float = 0.80   # 20% margin contraction
BASE_FCF_MARGIN_MULTIPLIER: float  = 1.00   # No adjustment
BULL_FCF_MARGIN_EXPANSION: float   = 1.20   # 20% margin expansion

# ─── Fallback / Default Values ────────────────────────────────────────────────
DEFAULT_BETA: float      = 1.5    # Reasonable for high-growth tech if yfinance is missing
DEFAULT_DEBT_COST: float = 0.05   # 5% pre-tax cost of debt fallback
DEFAULT_TAX_RATE: float  = 0.21   # U.S. statutory corporate tax rate

# Max Stage 1 growth rate cap — prevents runaway DCF outputs for hyper-growth names
MAX_GROWTH_CAP: float = 0.60      # 60%

# ─── Chart Styling ────────────────────────────────────────────────────────────
CHART_STYLE: str   = "dark_background"
CHART_FIGSIZE: tuple = (13, 7)
