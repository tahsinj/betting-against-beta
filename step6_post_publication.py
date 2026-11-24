# Step 6: out-of-sample check. Does BAB still work after FP's sample ends
# (April 2012 to 2025)? FP was circulated from 2010 and published in 2014, so this
# period is both out of sample and mostly after publication.

import matplotlib.pyplot as plt
import pandas as pd

from helpers import (BLUE, FP_END, GREY, NV_START, ORANGE, POST_END, POST_START,
                     load_aqr_bab, load_bab, load_deciles, load_factors, plot_style,
                     portfolio_table, regress, save_fig, save_table, strategy_table, subsample)

factors = load_factors()
aqr = load_aqr_bab()
bab = load_bab("bab")
bab_value = load_bab("bab_value")
bab_cap = load_bab("bab_rank_cap")

strategies = {
    "BAB (ours)": bab["bab"],
    "BAB net of costs": bab["bab"] - bab["cost"],
    "BAB value weighted terciles": bab_value["bab"],
    "BAB rank x cap weighted": bab_cap["bab"],
    "BAB rank x cap weighted, net": bab_cap["bab"] - bab_cap["cost"],
    "AQR BAB": aqr,
}

# --- before vs after publication -------------------------------------------------

tables = []
for period, (start, end) in {"1968-2012": (NV_START, FP_END), "2012-2025": (POST_START, POST_END)}.items():
    t = strategy_table({k: subsample(v, start, end) for k, v in strategies.items()}, factors)
    t.index = pd.MultiIndex.from_product([[period], t.index])
    tables.append(t)
save_table(pd.concat(tables), "post_table1_before_after")

# test whether the mean fell: regress the return on a constant and a dummy for
# months after FP's sample
rows = {}
for name, r in strategies.items():
    r = subsample(r, NV_START, POST_END).dropna()
    post = pd.Series((r.index >= pd.Period(POST_START, "M")).astype(float), index=r.index, name="post")
    fit = regress(r, post)
    rows[name] = {"mean before (%)": fit.params["const"] * 100,
                  "change after (%)": fit.params["post"] * 100, "t(change)": fit.tvalues["post"]}
save_table(pd.DataFrame(rows).T.round(2), "post_table2_change_in_mean")

# --- beta deciles after FP's sample --------------------------------------------------

for weighting in ["ew", "vw"]:
    dec_ret, dec_beta = load_deciles(weighting)
    excess = dec_ret.sub(factors["rf"], axis=0)
    excess.columns = [f"P{d}" for d in excess.columns]
    dec_beta.columns = [f"P{d}" for d in dec_beta.columns]
    excess["BAB"] = bab["bab"]
    save_table(portfolio_table(subsample(excess, POST_START, POST_END),
                               subsample(dec_beta, POST_START, POST_END), factors),
               f"post_table3_deciles_{weighting}_2012_2025")

# --- figure: cumulative returns since publication -----------------------------------

plot_style()
fig, ax = plt.subplots()
for name, color, style in [("BAB (ours)", BLUE, "-"), ("BAB net of costs", ORANGE, "-"),
                           ("BAB rank x cap weighted, net", GREY, ":"), ("AQR BAB", GREY, "--")]:
    r = subsample(strategies[name], POST_START, POST_END).dropna()
    ax.plot(r.index.to_timestamp(), (1 + r).cumprod(), color=color, linestyle=style, label=name)
ax.axhline(1, color=GREY, linewidth=0.8)
ax.set_ylabel("value of $1")
ax.set_title("BAB after FP's sample, April 2012 to 2025", loc="left")
ax.legend()
save_fig(fig, "fig5_post_publication")
