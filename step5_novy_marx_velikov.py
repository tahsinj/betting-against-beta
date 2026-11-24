# Step 5: the Novy-Marx and Velikov (2022) critique.
#
# Everything uses NV's published sample, January 1968 to December 2019, so our numbers can
# be compared with theirs.
#
#   Test 1  rank weights vs equal / value weighted tercile sorts
#   Test 2  how many BAB dollars sit in nano caps and in each NYSE size decile
#   Test 3  FP betas: OLS betas instead, forecast accuracy, bias in the market's
#           FP-beta, and BAB's beta at longer horizons
#   Test 4  returns net of transaction costs (Abdi-Ranaldo spreads, Gibbs as a check)
#   Test 5  is BAB spanned by profitability (RMW) and investment (CMA)?

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (BLUE, DATA_DIR, FP_START, GREY, NV_END, NV_START, ORANGE, POST_START,
                     alpha, load_bab, load_factors, plot_style, regress, regression_table,
                     save_fig, save_table, strategy_table, subsample)

factors = load_factors()
names = {"bab": "BAB (FP rank weights)", "bab_equal": "equal weighted terciles",
         "bab_value": "value weighted terciles", "bab_rank_cap": "rank x market cap weights",
         "bab_ex_nano": "BAB without bottom NYSE decile", "bab_ols": "BAB with daily OLS betas"}
series = {v: subsample(load_bab(v), NV_START, NV_END) for v in names}

# --- Test 1: does the weighting scheme drive BAB? ----------------------------------

weighting = ["bab", "bab_equal", "bab_value", "bab_rank_cap"]
test1 = strategy_table({names[v]: series[v]["bab"] for v in weighting}, factors,
                       models=("CAPM", "FF3", "FF5", "FF6"))
test1["corr with BAB"] = [series[v]["bab"].corr(series["bab"]["bab"]) for v in weighting]
save_table(test1.round(2), "nv_test1_weighting")

# --- Test 2: exposure to small stocks ------------------------------------------------

rows = {}
for v in ["bab", "bab_value", "bab_rank_cap", "bab_ex_nano"]:
    s = series[v]
    rows[names[v]] = {
        "long $": (1 / s["beta_L"]).mean(),    # each leg's weights sum to one
        "short $": (1 / s["beta_H"]).mean(),
        "$ in nano caps": s["nano_dollars"].mean(),
        "  of which long": s["nano_long"].mean(),
        "$ in bottom NYSE decile": s["bottom_decile_dollars"].mean(),
        "$ in micro caps (< NYSE p20)": s["micro_dollars"].mean(),
        "% of gross in micro caps": (s["micro_dollars"] / s["gross"]).mean() * 100,
    }
save_table(pd.DataFrame(rows).T.round(2), "nv_test2_small_stock_exposure")

# BAB holdings by NYSE size decile, cents per $1 of BAB (NV Table 1)
holdings = pd.read_parquet(os.path.join(DATA_DIR, "bab_holdings.parquet"))
holdings.index = pd.PeriodIndex(holdings["month"], freq="M")
holdings = subsample(holdings, NV_START, NV_END)
n_months = holdings.index.nunique()
by_decile = holdings.groupby("size_decile")[["long", "short"]].sum() / n_months * 100
by_decile.loc["total"] = by_decile.sum()
save_table(by_decile.round(1), "nv_test2b_holdings_by_size_decile")

# --- Test 3: FP betas ------------------------------------------------------------------

# (a) realized betas of the two legs vs their ex-ante betas. If the estimator is
# unbiased, realized and ex-ante should match and BAB should have zero market beta.
rows = {}
for v in ["bab", "bab_ols"]:
    s = series[v]
    rf = factors["rf"].reindex(s.index)
    fit = regress(s["bab"], factors["mktrf"])
    rows[names[v]] = {
        "beta_L ex ante": s["beta_L"].mean(),
        "beta_L realized": regress(s["r_L"] - rf, factors["mktrf"]).params["mktrf"],
        "beta_H ex ante": s["beta_H"].mean(),
        "beta_H realized": regress(s["r_H"] - rf, factors["mktrf"]).params["mktrf"],
        "BAB realized beta": fit.params["mktrf"], "t": fit.tvalues["mktrf"],
    }
save_table(pd.DataFrame(rows).T.round(2), "nv_test3a_leg_betas")

test3b = strategy_table({names[v]: series[v]["bab"] for v in ["bab", "bab_ols"]}, factors)
save_table(test3b, "nv_test3b_ols_beta_bab")

# (b) stock-level forecasts: how well does each beta predict the beta realized
# over the next 12 months? We use two yardsticks: the daily OLS beta (which is
# biased down for illiquid stocks, so it favours the daily OLS forecast) and an
# OLS beta on 3-day returns (which mostly removes that bias).
betas = pd.read_parquet(os.path.join(DATA_DIR, "betas.parquet"))
betas["month"] = pd.PeriodIndex(betas["month"], freq="M")
future = betas[["permno", "month", "beta_ols_raw", "beta_3day"]].copy()
future["month"] = future["month"] - 12
future = future.rename(columns={"beta_ols_raw": "realized_daily", "beta_3day": "realized_3day"})
fc = betas.merge(future, on=["permno", "month"])
fc = fc[(fc["month"] >= pd.Period(NV_START, "M")) & (fc["month"] <= pd.Period(NV_END, "M"))]
fc = fc.dropna(subset=["beta_fp", "beta_ts", "beta_ols", "beta_ols_raw", "realized_daily", "realized_3day"])

rows = {}
for col, label in {"beta_ts": "FP, not shrunk", "beta_fp": "FP, shrunk",
                   "beta_ols_raw": "OLS 1y daily, not shrunk", "beta_ols": "OLS 1y daily, shrunk"}.items():
    out = {"mean": fc[col].mean()}
    for realized in ["realized_3day", "realized_daily"]:
        err = fc[col] - fc[realized]
        # average cross-sectional slope of realized on predicted (1 = spread not too wide or narrow)
        slopes = fc.groupby("month").apply(lambda g: np.polyfit(g[col], g[realized], 1)[0])
        tag = "3-day" if realized == "realized_3day" else "daily"
        out[f"bias ({tag})"] = err.mean()
        out[f"RMSE ({tag})"] = np.sqrt((err ** 2).mean())
        out[f"slope ({tag})"] = slopes.mean()
    rows[label] = out
save_table(pd.DataFrame(rows).T.round(3), "nv_test3c_beta_forecasts")

# (c) the market's FP-beta. The market's true beta is always 1, so the value
# weighted average FP-beta across all stocks shows the bias in the estimator.
# NV find a mean of 1.05 that moves with the 1-year / 5-year market volatility ratio.
crsp = pd.read_parquet(os.path.join(DATA_DIR, "crsp_monthly.parquet"), columns=["permno", "date", "cap"])
crsp["month"] = crsp["date"].dt.to_period("M")
mk = betas.merge(crsp[["permno", "month", "cap"]], on=["permno", "month"]).dropna(subset=["beta_fp", "cap"])
mk["w"] = mk["cap"] / mk.groupby("month")["cap"].transform("sum")
market_fp_beta = (mk["w"] * mk["beta_fp"]).groupby(mk["month"]).sum().rename("market FP-beta")

ff = pd.read_parquet(os.path.join(DATA_DIR, "ff_daily.parquet")).set_index("date").sort_index()
mkt = np.log1p(ff["mktrf"] + ff["rf"]) - np.log1p(ff["rf"])
vol_ratio = np.log(mkt.rolling(252).std() / mkt.rolling(1260).std())
vol_ratio = vol_ratio.groupby(vol_ratio.index.to_period("M")).last().rename("log vol ratio")

mb = subsample(pd.concat([market_fp_beta, vol_ratio], axis=1).dropna(), NV_START, NV_END)
fit = regress(mb["market FP-beta"], mb["log vol ratio"], lags=60)
test3d = pd.Series({"mean market FP-beta": mb["market FP-beta"].mean(),
                    "std": mb["market FP-beta"].std(),
                    "slope on log(vol 1y / vol 5y)": fit.params["log vol ratio"],
                    "t (NW, 60 lags)": fit.tvalues["log vol ratio"],
                    "R2": fit.rsquared})
save_table(test3d.round(3).to_frame("value"), "nv_test3d_market_fp_beta")

# (d) BAB's beta at longer horizons (NV Table 4): small stocks react to the
# market with a delay, so lagged market returns pick up extra exposure. The
# interaction term checks whether BAB's beta moves with market volatility.
lags = pd.DataFrame({"mkt": factors["mktrf"], "mkt lag 1": factors["mktrf"].shift(1),
                     "mkt lag 2": factors["mktrf"].shift(2)})
z = (vol_ratio - vol_ratio.mean()) / vol_ratio.std()
lags["mkt x vol ratio"] = factors["mktrf"] * z.shift(1).reindex(factors.index)
bab = series["bab"]["bab"]
cols = {}
for label, X in {"(1)": ["mkt"], "(2)": ["mkt", "mkt lag 1", "mkt lag 2"],
                 "(3)": ["mkt", "mkt lag 1", "mkt lag 2", "mkt x vol ratio"]}.items():
    res = regress(bab, lags[X])
    coefs = res.params.copy()
    coefs["const"] *= 100   # intercept in % per month
    cols[label] = {k: f"{coefs[k]:.2f} ({res.tvalues[k]:.2f})" for k in coefs.index}
test3e = pd.DataFrame(cols).rename(index={"const": "intercept (%)"}).fillna("")
save_table(test3e, "nv_test3e_bab_lagged_beta")

# --- Test 4: transaction costs -----------------------------------------------------

rows = {}
for v in ["bab", "bab_ex_nano", "bab_rank_cap", "bab_value"]:
    s = series[v]
    gross = s["bab"]
    net = s["bab"] - s["cost"]   # cost of rebalancing at the start of the month
    ff5_net, ff5_t = alpha(net, factors, "FF5")
    rows[names[v]] = {
        "gross mean": gross.mean() * 100,
        # NV define turnover as the average of purchases and sales, per year
        "turnover (%/yr)": s["turnover"].mean() / 2 * 12 * 100,
        "cost": s["cost"].mean() * 100,
        "net mean": net.mean() * 100,
        "t(net)": net.mean() / net.std() * np.sqrt(len(net)),
        "% of gross lost": s["cost"].mean() / gross.mean() * 100,
        "net FF5 alpha": ff5_net * 100, "t(FF5)": ff5_t,
    }
save_table(pd.DataFrame(rows).T.round(2), "nv_test4_transaction_costs")

# robustness: the same costs using Hasbrouck's Gibbs estimates (what NV's cost
# method is built on). These only go up to 2009, so compare on 1968-2009.
rows = {}
for v in ["bab", "bab_ex_nano", "bab_rank_cap", "bab_value"]:
    s = subsample(series[v], NV_START, "2009-12")
    row = {"gross mean": s["bab"].mean() * 100}
    for col, label in [("cost", "Abdi-Ranaldo"), ("cost_gibbs", "Gibbs")]:
        net = s["bab"] - s[col]
        ff5_net, ff5_t = alpha(net, factors, "FF5")
        row[f"cost ({label})"] = s[col].mean() * 100
        row[f"net mean ({label})"] = net.mean() * 100
        row[f"net FF5 alpha ({label})"] = ff5_net * 100
        row[f"t ({label})"] = ff5_t
    rows[names[v]] = row
save_table(pd.DataFrame(rows).T.round(2), "nv_test4b_costs_gibbs_1968_2009")

# --- Test 5: spanning by profitability and investment ------------------------------

save_table(regression_table(series["bab"]["bab"], factors, ["CAPM", "FF3", "FF5", "FF6"]), "nv_test5_bab_ff6")
save_table(regression_table(series["bab_value"]["bab"], factors, ["CAPM", "FF3", "FF5", "FF6"]), "nv_test5b_value_bab_ff6")
save_table(regression_table(series["bab"]["bab"] - series["bab"]["cost"], factors, ["CAPM", "FF3", "FF5", "FF6"]),
           "nv_test5c_net_bab_ff6")

# --- Figures ---------------------------------------------------------------------------

plot_style()

# nano-cap dollars over the whole sample (12-month moving average)
full = subsample(load_bab("bab"), FP_START)
fig, ax = plt.subplots()
ax.plot(full.index.to_timestamp(), full["nano_dollars"].rolling(12).mean(), color=BLUE, label="nano caps (bottom 1% of market cap)")
ax.plot(full.index.to_timestamp(), full["micro_dollars"].rolling(12).mean(), color=ORANGE, label="micro caps (< NYSE 20th pct)")
ax.axvline(pd.Timestamp(POST_START), color=GREY, linestyle=":", linewidth=1)
ax.set_ylabel("dollars per $1 of BAB (12m avg)")
ax.set_title("BAB dollars held in the smallest stocks", loc="left")
ax.legend()
save_fig(fig, "fig3_nano_caps")

# gross vs net cumulative returns
s = series["bab"]
fig, ax = plt.subplots()
ax.plot(s.index.to_timestamp(), (1 + s["bab"]).cumprod(), color=BLUE, label="BAB, gross")
ax.plot(s.index.to_timestamp(), (1 + s["bab"] - s["cost"]).cumprod(), color=ORANGE, label="BAB, net of trading costs")
v = series["bab_rank_cap"]
ax.plot(v.index.to_timestamp(), (1 + v["bab"] - v["cost"]).cumprod(), color=GREY, linestyle="--", label="rank x cap weighted, net")
ax.set_yscale("log")
ax.set_ylabel("value of $1 (log scale)")
ax.set_title(f"BAB before and after trading costs, {NV_START[:4]}-{NV_END[:4]}", loc="left")
ax.legend()
save_fig(fig, "fig4_net_of_costs")

# the market's FP-beta and the market volatility ratio (NV Figure 5)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
t = mb.index.to_timestamp()
ax1.plot(t, mb["market FP-beta"], color=BLUE)
ax1.axhline(1, color=GREY, linewidth=0.8)
ax1.set_ylabel("FP-beta of market")
ax1.set_title("The market's true beta is 1, its FP-beta is not", loc="left")
ax2.plot(t, mb["log vol ratio"], color=ORANGE)
ax2.axhline(0, color=GREY, linewidth=0.8)
ax2.set_ylabel("log(1y / 5y vol)")
save_fig(fig, "fig6_market_fp_beta")
