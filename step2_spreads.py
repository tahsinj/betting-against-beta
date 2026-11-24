# Step 2: monthly bid-ask spread estimates from daily high, low and close prices
# using Abdi and Ranaldo (2017).
#
# With c = log close and eta = (log high + log low) / 2, the spread satisfies
#   s^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})]
# We average the product over consecutive days within each stock-month and set
# negative estimates to zero (their "monthly" estimator).
#
# Outputs:
#   data/spreads.parquet       (permno, month, spread), spread as a fraction of price
#   data/gibbs_costs.parquet   (permno, year, gibbs_cost), Hasbrouck's estimates up to 2009

import glob
import os

import numpy as np
import pandas as pd

from helpers import DATA_DIR

MIN_DAYS = 10   # need at least this many day pairs in a month

results = []
for f in sorted(glob.glob(os.path.join(DATA_DIR, "crsp_daily", "*.parquet"))):
    d = pd.read_parquet(f, columns=["permno", "date", "prc", "high", "low", "vol"])
    d["prc"] = d["prc"].abs()   # negative price = bid/ask midpoint in old CRSP format

    # drop no-trade days (high/low are then quotes, not trades) and bad ranges;
    # days with missing volume are kept
    ok = (d["vol"] != 0) & (d["low"] > 0) & (d["high"] >= d["low"]) \
        & (d["prc"] >= d["low"]) & (d["prc"] <= d["high"])
    d = d[ok].sort_values(["permno", "date"])
    if len(d) == 0:
        print(os.path.basename(f), "no usable high/low data")
        continue

    d["month"] = d["date"].dt.to_period("M")
    d["c"] = np.log(d["prc"])
    d["eta"] = (np.log(d["high"]) + np.log(d["low"])) / 2
    # next trading day's mid-range, only inside the same stock and month
    d["eta_next"] = d.groupby(["permno", "month"])["eta"].shift(-1)
    d["product"] = (d["c"] - d["eta"]) * (d["c"] - d["eta_next"])

    monthly = d.groupby(["permno", "month"])["product"].agg(["mean", "count"]).reset_index()
    monthly = monthly[monthly["count"] >= MIN_DAYS]
    monthly["spread"] = np.sqrt(np.maximum(4 * monthly["mean"], 0))
    results.append(monthly[["permno", "month", "spread"]])
    print(os.path.basename(f), f"{len(monthly):,} stock-months, median spread {monthly['spread'].median():.4f}")

spreads = pd.concat(results, ignore_index=True)
spreads["month"] = spreads["month"].astype(str)
spreads.to_parquet(os.path.join(DATA_DIR, "spreads.parquet"), index=False)

# coverage by decade, to check when high/low data becomes reliable
spreads["decade"] = spreads["month"].str[:3] + "0s"
print("\nstock-months with a spread estimate, by decade:")
print(spreads.groupby("decade")["spread"].agg(["count", "median", "mean"]).round(4))

# --- Hasbrouck (2009) Gibbs effective costs, used as a robustness check -------------
# NV's cost method starts from these. Hasbrouck posts one estimate per stock-year
# (up to 2009) at https://pages.stern.nyu.edu/~jhasbrou/Research/GibbsCurrent/
# His c is the effective cost, which is already half the spread.
gibbs_cols = ["permno", "year", "kSample", "c", "beta", "varu", "sdu", "exchcd", "shrcd",
              "cfacpr", "firstDate", "lastDate", "nDays", "nTradeDays"]
gibbs = pd.read_csv(os.path.join(DATA_DIR, "crspGibbs2009v01.txt"), sep=r"\s+", header=None,
                    names=gibbs_cols, na_values=".")
# a few stock-years are split into more than one sample (kSample), so average them
gibbs = gibbs.dropna(subset=["c"]).groupby(["permno", "year"])["c"].mean()
gibbs = gibbs.rename("gibbs_cost").reset_index()
gibbs.to_parquet(os.path.join(DATA_DIR, "gibbs_costs.parquet"), index=False)
print(f"\nGibbs effective costs: {len(gibbs):,} stock-years, {gibbs['year'].min()}-{gibbs['year'].max()}, "
      f"median {gibbs['gibbs_cost'].median():.4f}")
