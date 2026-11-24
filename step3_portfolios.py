# Step 3: build the BAB factor (and the variants needed for the Novy-Marx &
# Velikov tests) plus beta-sorted decile portfolios.
#
# At the end of month t we sort stocks on beta estimated with data up to t and
# hold the portfolios over month t+1. Monthly CRSP returns already include
# delisting returns.
#
# BAB construction (FP eq. 16-17):
#   z = rank of beta, w_H = k (z - zbar)^+, w_L = k (z - zbar)^-, k = 2 / sum|z - zbar|
#   r_BAB = (r_L - rf) / beta_L - (r_H - rf) / beta_H
#
# Outputs:
#   data/bab_series.parquet  one row per (variant, month): BAB return, leg returns
#                            and betas, small-stock exposure, turnover, costs
#   data/deciles.parquet     one row per (decile, month), equal and value weighted
#   data/bab_holdings.parquet  BAB long/short dollars by NYSE size decile each month

import os

import numpy as np
import pandas as pd

from helpers import DATA_DIR, load_factors

# --- monthly stock panel -------------------------------------------------------

crsp = pd.read_parquet(os.path.join(DATA_DIR, "crsp_monthly.parquet"))
crsp["month"] = crsp["date"].dt.to_period("M")
crsp = crsp[crsp["cap"] > 0].drop_duplicates(["permno", "month"])

# return over the next month (what the portfolio formed at t earns)
nxt = crsp[["permno", "month", "ret"]].copy()
nxt["month"] = nxt["month"] - 1
crsp = crsp.merge(nxt.rename(columns={"ret": "ret_next"}), on=["permno", "month"], how="left")

# NYSE size breakpoints (10th, 20th, ..., 90th percentile) every month
nyse_size_breaks = crsp[crsp["primaryexch"] == "N"].groupby("month")["cap"] \
    .quantile(np.arange(0.1, 1.0, 0.1)).unstack()
crsp["micro"] = crsp["cap"] < crsp["month"].map(nyse_size_breaks.iloc[:, 1])          # below NYSE 20th pct
crsp["bottom_decile"] = crsp["cap"] < crsp["month"].map(nyse_size_breaks.iloc[:, 0])  # below NYSE 10th pct

# nano caps (NV's definition): the smallest stocks that together make up 1% of
# total market cap
crsp = crsp.sort_values(["month", "cap"])
cum_share = crsp.groupby("month")["cap"].cumsum() / crsp.groupby("month")["cap"].transform("sum")
crsp["nano"] = cum_share <= 0.01

# spreads: fill missing ones with the median of stocks of similar size that month
spreads = pd.read_parquet(os.path.join(DATA_DIR, "spreads.parquet"))
spreads["month"] = pd.PeriodIndex(spreads["month"], freq="M")
crsp = crsp.merge(spreads, on=["permno", "month"], how="left")
crsp["size_decile"] = crsp.groupby("month")["cap"].transform(
    lambda c: pd.qcut(c.rank(method="first"), 10, labels=False))
crsp["spread"] = crsp["spread"].fillna(crsp.groupby(["month", "size_decile"])["spread"].transform("median"))
crsp["spread"] = crsp["spread"].fillna(crsp.groupby("month")["spread"].transform("median"))

# Hasbrouck's Gibbs effective cost (already a half spread), only available up to
# 2009. Used for a robustness check on the costs; filled the same way.
gibbs = pd.read_parquet(os.path.join(DATA_DIR, "gibbs_costs.parquet"))
crsp["year"] = crsp["month"].dt.year
crsp = crsp.merge(gibbs, on=["permno", "year"], how="left")
crsp["gibbs_cost"] = crsp["gibbs_cost"].fillna(crsp.groupby(["month", "size_decile"])["gibbs_cost"].transform("median"))

betas = pd.read_parquet(os.path.join(DATA_DIR, "betas.parquet"), columns=["permno", "month", "beta_fp", "beta_ols"])
betas["month"] = pd.PeriodIndex(betas["month"], freq="M")
crsp = crsp.merge(betas, on=["permno", "month"], how="left")

rf = load_factors()["rf"]


# --- portfolio functions ---------------------------------------------------------

def leg_weights(beta, cap, scheme):
    """Weights of the low-beta and high-beta legs, each summing to one.
    rank     = FP rank weights over all stocks (dividing by the sum is the same as FP's k)
    rank_cap = rank weights times market cap (NV footnote 6)
    equal    = equal weights in the bottom and top thirds of beta (NV section 2.1)
    value    = value weights in the bottom and top thirds of beta"""
    if scheme in ("rank", "rank_cap"):
        z = beta.rank()
        dev = z - z.mean()
        w_low = (-dev).clip(lower=0)
        w_high = dev.clip(lower=0)
        if scheme == "rank_cap":
            w_low, w_high = w_low * cap, w_high * cap
    else:
        size = cap if scheme == "value" else pd.Series(1.0, index=beta.index)
        w_low = size.where(beta <= beta.quantile(1 / 3), 0)
        w_high = size.where(beta >= beta.quantile(2 / 3), 0)
    return w_low / w_low.sum(), w_high / w_high.sum()


def drift(pos, ret):
    """Positions after one month of returns. Weights inside each leg move with
    the stock returns while the dollar size of each leg stays the same."""
    grown = pos * (1 + ret)
    long_, short = grown[pos > 0], grown[pos < 0]
    long_ = long_ / long_.sum() * pos[pos > 0].sum()
    short = short / short.sum() * pos[pos < 0].sum()
    return pd.concat([long_, short])


# variant name -> (beta column, weighting scheme, drop bottom NYSE size decile?)
VARIANTS = {
    "bab": ("beta_fp", "rank", False),            # Frazzini-Pedersen BAB
    "bab_equal": ("beta_fp", "equal", False),     # NV test 1
    "bab_value": ("beta_fp", "value", False),     # NV test 1
    "bab_rank_cap": ("beta_fp", "rank_cap", False),  # NV's "implementable" BAB
    "bab_ex_nano": ("beta_fp", "rank", True),     # NV: BAB without the smallest NYSE decile
    "bab_ols": ("beta_ols", "rank", False),       # NV test 3
}

# --- monthly loop ------------------------------------------------------------------

rows, decile_rows, holding_rows = [], [], []
prev = {name: None for name in VARIANTS}   # last month's positions and returns, for turnover

for month, df in crsp.groupby("month"):
    if month + 1 not in rf.index or month not in nyse_size_breaks.index:
        continue
    df = df.set_index("permno")
    rf_next = rf[month + 1]
    universe = df.dropna(subset=["beta_fp", "beta_ols", "ret_next"])
    if len(universe) < 20:
        continue

    for name, (beta_col, scheme, drop_small) in VARIANTS.items():
        u = universe[~universe["bottom_decile"]] if drop_small else universe
        beta = u[beta_col]
        w_low, w_high = leg_weights(beta, u["cap"], scheme)

        beta_L, beta_H = (w_low * beta).sum(), (w_high * beta).sum()
        r_L, r_H = (w_low * u["ret_next"]).sum(), (w_high * u["ret_next"]).sum()
        bab = (r_L - rf_next) / beta_L - (r_H - rf_next) / beta_H

        # dollar position in each stock per $1 of BAB (long low beta, short high beta)
        pos = w_low / beta_L - w_high / beta_H
        pos = pos[pos != 0]
        held = u.loc[pos.index]

        # trading needed to go from last month's (drifted) positions to the new ones,
        # each dollar traded costs half the spread
        turnover, cost, cost_gibbs = np.nan, np.nan, np.nan
        if prev[name] is not None:
            old_pos, old_ret = prev[name]
            trades = pos.sub(drift(old_pos, old_ret), fill_value=0).abs()
            half_spread = df["spread"].reindex(trades.index) / 2
            half_spread = half_spread.fillna(half_spread.median())
            turnover = trades.sum()
            cost = (trades * half_spread).sum(min_count=1)
            gibbs_cost = df["gibbs_cost"].reindex(trades.index)
            if gibbs_cost.notna().any():   # Hasbrouck's estimates stop in 2009
                cost_gibbs = (trades * gibbs_cost.fillna(gibbs_cost.median())).sum()
        prev[name] = (pos, held["ret_next"])

        rows.append({
            "variant": name, "month": month + 1, "bab": bab,
            "r_L": r_L, "r_H": r_H, "beta_L": beta_L, "beta_H": beta_H,
            "n_stocks": len(u),
            "gross": pos.abs().sum(),
            "nano_dollars": pos[held["nano"]].abs().sum(),
            "nano_long": pos[held["nano"] & (pos > 0)].sum(),
            "bottom_decile_dollars": pos[held["bottom_decile"]].abs().sum(),
            "micro_dollars": pos[held["micro"]].abs().sum(),
            "turnover": turnover, "cost": cost, "cost_gibbs": cost_gibbs,
        })

        # where the FP BAB dollars sit, by NYSE size decile (NV Table 1)
        if name == "bab":
            size_dec = np.searchsorted(nyse_size_breaks.loc[month].values, held["cap"].values, side="right") + 1
            by_dec = pd.DataFrame({"dec": size_dec, "long": pos.clip(lower=0), "short": -pos.clip(upper=0)})
            for d, g in by_dec.groupby("dec"):
                holding_rows.append({"month": month + 1, "size_decile": d,
                                     "long": g["long"].sum(), "short": g["short"].sum()})

    # beta deciles with NYSE breakpoints (FP Table 3 is equal weighted)
    nyse_betas = universe.loc[universe["primaryexch"] == "N", "beta_fp"]
    beta_breaks = nyse_betas.quantile(np.arange(0.1, 1.0, 0.1)).values
    decile = np.searchsorted(beta_breaks, universe["beta_fp"].values, side="right") + 1
    for d, g in universe.groupby(decile):
        w = g["cap"] / g["cap"].sum()
        decile_rows.append({"decile": d, "month": month + 1,
                            "ret_ew": g["ret_next"].mean(), "ret_vw": (w * g["ret_next"]).sum(),
                            "beta_ew": g["beta_fp"].mean(), "beta_vw": (w * g["beta_fp"]).sum(),
                            "n_stocks": len(g)})

    if month.month == 12:
        print(month.year, f"{len(universe):,} stocks")

for out, name in [(rows, "bab_series"), (decile_rows, "deciles"), (holding_rows, "bab_holdings")]:
    out = pd.DataFrame(out)
    out["month"] = out["month"].astype(str)
    out.to_parquet(os.path.join(DATA_DIR, name + ".parquet"), index=False)

bab_series = pd.DataFrame(rows)
print("\naverage by variant:")
print(bab_series.groupby("variant")[["bab", "beta_L", "beta_H", "gross", "nano_dollars", "turnover", "cost"]].mean().round(4))
