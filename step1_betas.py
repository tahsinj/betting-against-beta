# Step 1: estimate betas at every month end from CRSP daily returns.
#
# Frazzini-Pedersen (2014) beta:
#   beta_TS = corr(i, m) * vol_i / vol_m
#   vol: 1-year rolling std of daily log excess returns (at least 120 days)
#   corr: 5-year rolling correlation of overlapping 3-day log excess returns
#         (at least 750 days), the 3-day returns deal with nonsynchronous trading
#   beta_FP = 0.6 * beta_TS + 0.4 * 1   (shrink towards the cross-sectional mean)
#
# For the Novy-Marx & Velikov comparison we also keep a plain OLS beta from a
# 1-year regression of daily returns on the market (at least 120 days).
#
# Output: data/betas.parquet with one row per stock-month.

import glob
import os

import numpy as np
import pandas as pd

from helpers import DATA_DIR

VOL_WINDOW, VOL_MIN = 252, 120
CORR_WINDOW, CORR_MIN = 1260, 750
SHRINK = 0.6

# market = FF market return (CRSP value-weighted index), in log excess returns
ff = pd.read_parquet(os.path.join(DATA_DIR, "ff_daily.parquet")).set_index("date").sort_index()
log_rf = np.log1p(ff["rf"])
mkt = np.log1p(ff["mktrf"] + ff["rf"]) - log_rf
trading_days = mkt.index

mkt_vol = mkt.rolling(VOL_WINDOW, min_periods=VOL_MIN).std()
mkt_3day = mkt.rolling(3).sum()

# last trading day of every month, this is where we sample the betas
month_ends = trading_days.to_series().groupby(trading_days.to_period("M")).max()

print("loading daily CRSP...")
files = sorted(glob.glob(os.path.join(DATA_DIR, "crsp_daily", "*.parquet")))
daily = pd.concat([pd.read_parquet(f, columns=["permno", "date", "ret"]) for f in files])
daily = daily[daily["date"] >= trading_days[0]]   # FF daily data starts July 1926
daily = daily.sort_values(["permno", "date"])
print(f"{len(daily):,} rows, {daily['permno'].nunique():,} stocks")

results = []
for n, (permno, g) in enumerate(daily.groupby("permno", sort=False)):
    if n % 2000 == 0:
        print(f"  stock {n:,}")
    if g["ret"].count() < VOL_MIN:
        continue

    # put the stock on the market's trading calendar so that the rolling windows
    # are in trading days (days with no CRSP record become NaN)
    days = trading_days[(trading_days >= g["date"].iloc[0]) & (trading_days <= g["date"].iloc[-1])]
    ret = g.set_index("date")["ret"].reindex(days)
    x = np.log1p(ret) - log_rf.reindex(days)
    m = mkt.reindex(days)

    x3, m3 = x.rolling(3).sum(), mkt_3day.reindex(days)
    vol = x.rolling(VOL_WINDOW, min_periods=VOL_MIN).std()
    corr = x3.rolling(CORR_WINDOW, min_periods=CORR_MIN).corr(m3)

    # OLS beta = cov / var, using only days where the stock has a return
    m_same_days = m.where(x.notna())
    cov = x.rolling(VOL_WINDOW, min_periods=VOL_MIN).cov(m_same_days)
    var = m_same_days.rolling(VOL_WINDOW, min_periods=VOL_MIN).var()

    # 1-year OLS beta on 3-day returns, only used as a "realized beta" yardstick
    # in step 5 because daily betas are biased down for thinly traded stocks
    m3_same_days = m3.where(x3.notna())
    beta_3day = x3.rolling(VOL_WINDOW, min_periods=VOL_MIN).cov(m3_same_days) \
        / m3_same_days.rolling(VOL_WINDOW, min_periods=VOL_MIN).var()

    ends = month_ends[(month_ends >= days[0]) & (month_ends <= days[-1])]
    out = pd.DataFrame({
        "permno": permno,
        "month": ends.index,
        "vol": vol.loc[ends].values,
        "corr": corr.loc[ends].values,
        "beta_ols_raw": (cov / var).loc[ends].values,
        "beta_3day": beta_3day.loc[ends].values,
    })
    results.append(out.dropna(how="all", subset=["vol", "corr", "beta_ols_raw"]))

betas = pd.concat(results, ignore_index=True)
mkt_vol_monthly = pd.Series(mkt_vol.loc[month_ends].values, index=month_ends.index)
betas["mkt_vol"] = betas["month"].map(mkt_vol_monthly)

betas["beta_ts"] = betas["corr"] * betas["vol"] / betas["mkt_vol"]
betas["beta_fp"] = SHRINK * betas["beta_ts"] + (1 - SHRINK) * 1.0
# shrink the OLS beta the same way so the two only differ in the estimator
betas["beta_ols"] = SHRINK * betas["beta_ols_raw"] + (1 - SHRINK) * 1.0

betas["month"] = betas["month"].astype(str)   # parquet can't store Periods
betas.to_parquet(os.path.join(DATA_DIR, "betas.parquet"), index=False)

print(f"\nsaved {len(betas):,} stock-months")
print(betas[["beta_ts", "beta_fp", "beta_ols_raw", "beta_ols"]].describe().round(3))
