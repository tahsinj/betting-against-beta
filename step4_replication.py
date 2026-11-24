# Step 4: replicate the main US results of Frazzini and Pedersen (2014) over
# their sample (1931 to March 2012) and compare our BAB factor with AQR's.
#
# Tables: beta deciles + BAB (FP Table 3, equal weighted deciles), the same with
#         value weighted deciles, BAB factor regressions, AQR comparison
# Figures: decile alphas and betas, cumulative BAB returns vs AQR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (BLUE, FP_END, FP_START, GREY, ORANGE, POST_START, load_aqr_bab,
                     load_bab, load_deciles, load_factors, plot_style, portfolio_table, regress,
                     regression_table, save_fig, save_table, subsample, summary_stats)

factors = load_factors()
bab = load_bab("bab")
aqr = load_aqr_bab()

# --- Table 1: beta-sorted deciles and BAB ----------------------------------------
# FP's Table 3 uses equal weighted deciles with NYSE breakpoints. We also show
# value weighted deciles because NV argue the results depend on small stocks.

for weighting, name in [("ew", "table1_deciles_1931_2012"), ("vw", "table1b_deciles_vw_1931_2012")]:
    dec_ret, dec_beta = load_deciles(weighting)
    excess = dec_ret.sub(factors["rf"], axis=0)
    excess.columns = [f"P{d}" for d in excess.columns]
    dec_beta.columns = [f"P{d}" for d in dec_beta.columns]
    excess["BAB"] = bab["bab"]
    table = portfolio_table(subsample(excess, FP_START, FP_END), subsample(dec_beta, FP_START, FP_END), factors)
    save_table(table, name)
    if weighting == "ew":
        table1 = table

# --- Table 2: BAB regressions on the factor models FP use ------------------------

bab_fp = subsample(bab["bab"], FP_START, FP_END)
table2 = regression_table(bab_fp, factors, ["CAPM", "FF3", "Carhart", "FP5"])
save_table(table2, "table2_bab_regressions_1931_2012")

# leg characteristics, FP report these alongside the factor
legs = subsample(bab, FP_START, FP_END)[["beta_L", "beta_H", "n_stocks"]].mean()
legs["leverage long (1/beta_L)"] = (1 / subsample(bab, FP_START, FP_END)["beta_L"]).mean()
legs["leverage short (1/beta_H)"] = (1 / subsample(bab, FP_START, FP_END)["beta_H"]).mean()
save_table(legs.round(2).to_frame("average"), "table2b_bab_legs_1931_2012")

# --- Table 3: our BAB vs AQR's BAB --------------------------------------------------

both = pd.concat([bab["bab"].rename("ours"), aqr], axis=1).dropna()
rows = {}
for label, (start, end) in {"1931-2012": (FP_START, FP_END),
                            "2012-2025": (POST_START, None),
                            "full sample": (FP_START, None)}.items():
    b = subsample(both, start, end)
    ours, theirs = summary_stats(b["ours"]), summary_stats(b["aqr_bab"])
    fit = regress(b["ours"], b["aqr_bab"])
    rows[label] = {
        "correlation": b["ours"].corr(b["aqr_bab"]),
        "ours: mean (%/month)": ours["mean"] * 100, "ours: Sharpe": ours["sharpe"],
        "AQR: mean (%/month)": theirs["mean"] * 100, "AQR: Sharpe": theirs["sharpe"],
        "slope on AQR": fit.params["aqr_bab"], "intercept (%/month)": fit.params["const"] * 100,
        "months": len(b),
    }
table3 = pd.DataFrame(rows).T.round(3)
table3["months"] = table3["months"].astype(int)
save_table(table3, "table3_aqr_comparison")

# --- Figures -------------------------------------------------------------------------

plot_style()

# Figure 1: alphas fall with beta while realized betas line up with ex-ante betas
deciles = table1.drop(columns="BAB")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
x = np.arange(1, 11)
ax1.bar(x, deciles.loc["CAPM alpha"], color=BLUE, width=0.7)
ax1.axhline(0, color=GREY, linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xlabel("beta decile (1 = low)")
ax1.set_ylabel("CAPM alpha (% per month)")
ax1.set_title("Alpha by beta decile (equal weighted), 1931-2012", loc="left", fontsize=9)

ax2.plot(x, deciles.loc["beta (ex ante)"], "o-", color=BLUE, label="ex-ante beta")
ax2.plot(x, deciles.loc["beta (realized)"], "s--", color=ORANGE, label="realized beta")
ax2.set_xticks(x)
ax2.set_xlabel("beta decile (1 = low)")
ax2.set_ylabel("beta")
ax2.set_title("Ex-ante vs realized beta", loc="left")
ax2.legend()
save_fig(fig, "fig1_deciles")

# Figure 2: growth of $1 in our BAB and AQR's BAB (log scale)
fig, ax = plt.subplots()
growth = (1 + subsample(both, FP_START)).cumprod()
ax.plot(growth.index.to_timestamp(), growth["ours"], color=BLUE, label="our BAB")
ax.plot(growth.index.to_timestamp(), growth["aqr_bab"], color=ORANGE, label="AQR BAB")
ax.set_yscale("log")
ax.axvline(pd.Timestamp(POST_START), color=GREY, linestyle=":", linewidth=1)
ax.text(pd.Timestamp(POST_START), ax.get_ylim()[0] * 1.5, " end of FP sample", color=GREY, fontsize=8)
ax.set_ylabel("value of $1 (log scale)")
ax.set_title("Cumulative BAB returns", loc="left")
ax.legend()
save_fig(fig, "fig2_bab_vs_aqr")

print("\ncorrelation with AQR (1931-2012):", round(table3.loc["1931-2012", "correlation"], 3))
