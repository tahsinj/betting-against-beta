# Shared settings and small functions used by the step scripts.

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = "data"
TABLE_DIR = os.path.join("output", "tables")
FIG_DIR = os.path.join("output", "figures")

# sample periods
FP_START, FP_END = "1931-01", "2012-03"      # FP's US sample ends March 2012 (betas need ~3 years first)
NV_START, NV_END = "1968-01", "2019-12"      # Novy-Marx & Velikov (2022) sample
POST_START, POST_END = "2012-04", "2025-12"  # everything after FP's sample

# Newey-West lags for monthly data: floor(4 * (T/100)^(2/9)) is about 6 for T ~ 1000
NW_LAGS = 6

# factor models used for alphas
MODELS = {
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "Carhart": ["mktrf", "smb", "hml", "umd"],
    "FP5": ["mktrf", "smb", "hml", "umd", "liq"],
    "FF5": ["mktrf", "smb5", "hml5", "rmw", "cma"],
    "FF6": ["mktrf", "smb5", "hml5", "rmw", "cma", "umd"],
}


def load_factors():
    """Monthly factors indexed by month (Period). FF5 versions of SMB/HML are
    kept separately as smb5/hml5 because they are built differently."""
    ff3 = pd.read_parquet(os.path.join(DATA_DIR, "ff3_monthly.parquet"))
    ff5 = pd.read_parquet(os.path.join(DATA_DIR, "ff5_monthly.parquet"))
    ps = pd.read_parquet(os.path.join(DATA_DIR, "ps_liq_monthly.parquet"))

    # WRDS dates are not all month-end, so join everything on year-month
    ff3["month"] = ff3["date"].dt.to_period("M")
    ff5["month"] = ff5["date"].dt.to_period("M")
    ps["month"] = ps["date"].dt.to_period("M")

    ff5 = ff5.rename(columns={"smb": "smb5", "hml": "hml5"})
    ps = ps.rename(columns={"ps_vwf": "liq"})
    # PS uses -99 for months before the traded factor exists
    ps.loc[ps["liq"] <= -99, "liq"] = np.nan

    factors = ff3.set_index("month")[["mktrf", "smb", "hml", "umd", "rf"]]
    factors = factors.join(ff5.set_index("month")[["smb5", "hml5", "rmw", "cma"]])
    factors = factors.join(ps.set_index("month")[["liq"]])
    return factors


def load_aqr_bab():
    """AQR's US BAB factor (monthly, decimal returns) indexed by month."""
    aqr = pd.read_excel(os.path.join(DATA_DIR, "aqr_bab_monthly.xlsx"),
                        sheet_name="BAB Factors", header=18)
    aqr = aqr[["DATE", "USA"]].dropna()
    aqr["month"] = pd.to_datetime(aqr["DATE"]).dt.to_period("M")
    return aqr.set_index("month")["USA"].rename("aqr_bab")


def load_bab(variant="bab"):
    """Monthly series for one BAB variant from step 3, indexed by return month."""
    s = pd.read_parquet(os.path.join(DATA_DIR, "bab_series.parquet"))
    s = s[s["variant"] == variant].copy()
    s.index = pd.PeriodIndex(s["month"], freq="M")
    return s.drop(columns=["variant", "month"])


def load_deciles(weighting="ew"):
    """Decile returns and ex-ante betas as two (month x decile) tables.
    weighting is "ew" (equal, as in FP Table 3) or "vw" (value weighted)."""
    d = pd.read_parquet(os.path.join(DATA_DIR, "deciles.parquet"))
    d["month"] = pd.PeriodIndex(d["month"], freq="M")
    rets = d.pivot(index="month", columns="decile", values="ret_" + weighting)
    betas = d.pivot(index="month", columns="decile", values="beta_" + weighting)
    return rets, betas


def regress(y, X, lags=NW_LAGS):
    """OLS of y on X (plus a constant) with Newey-West standard errors.
    Months where anything is missing are dropped."""
    data = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def alpha(y, factors, model):
    """Returns (alpha, t-stat) for one factor model."""
    res = regress(y, factors[MODELS[model]])
    return res.params["const"], res.tvalues["const"]


def summary_stats(r):
    """Mean, t-stat, volatility and Sharpe ratio of a monthly excess return series."""
    r = r.dropna()
    mean = r.mean()
    t = mean / (r.std() / np.sqrt(len(r)))
    return {"mean": mean, "t": t, "vol": r.std() * np.sqrt(12),
            "sharpe": mean / r.std() * np.sqrt(12), "months": len(r)}


def regression_table(y, factors, models):
    """Coefficients with Newey-West t-stats underneath, one column per model.
    Alpha is reported in % per month, factor loadings as they are."""
    rows = ["const"] + list(dict.fromkeys(f for m in models for f in MODELS[m]))
    table = {}
    for m in models:
        res = regress(y, factors[MODELS[m]])
        col = []
        for name in rows:
            if name in res.params:
                coef = res.params[name] * 100 if name == "const" else res.params[name]
                col += [f"{coef:.2f}", f"({res.tvalues[name]:.2f})"]
            else:
                col += ["", ""]
        col += [f"{res.rsquared:.2f}", f"{int(res.nobs)}"]
        table[m] = col

    index = []
    for name in rows:
        index += ["alpha" if name == "const" else name, ""]
    index += ["R2", "months"]
    return pd.DataFrame(table, index=index)


def portfolio_table(excess, exante_beta, factors):
    """FP Table 3 style: one column per portfolio. Returns and alphas in % per month."""
    rows = {}
    for col in excess.columns:
        r = excess[col].dropna()
        s = summary_stats(r)
        out = {"excess return": s["mean"] * 100, "  t": s["t"]}
        for m in ["CAPM", "FF3", "Carhart", "FP5"]:
            a, t = alpha(r, factors, m)
            out[f"{m} alpha"] = a * 100
            out[f"  t ({m})"] = t
        out["beta (ex ante)"] = exante_beta[col].mean() if col in exante_beta else 0.0
        out["beta (realized)"] = regress(r, factors["mktrf"]).params["mktrf"]
        out["volatility (%/yr)"] = s["vol"] * 100
        out["Sharpe ratio"] = s["sharpe"]
        rows[col] = out
    return pd.DataFrame(rows).round(2)


def strategy_table(strategies, factors, models=("CAPM", "Carhart", "FF6")):
    """One row per strategy: mean, t-stat, Sharpe and alphas (% per month)."""
    rows = {}
    for name, r in strategies.items():
        r = r.dropna()
        s = summary_stats(r)
        out = {"mean": s["mean"] * 100, "t": s["t"], "sharpe": s["sharpe"]}
        for m in models:
            a, t = alpha(r, factors, m)
            out[f"{m} alpha"] = a * 100
            out[f"t({m})"] = t
        out["months"] = s["months"]
        rows[name] = out
    table = pd.DataFrame(rows).T.round(2)
    table["months"] = table["months"].astype(int)
    return table


def save_table(df, name, index=True):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=index)
    df.to_latex(os.path.join(TABLE_DIR, name + ".tex"), index=index,
                float_format="%.2f", escape=True)
    print(f"\n{name}\n{df.to_string()}")


BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a897f"


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (7, 3.8), "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.grid": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    })
    os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name):
    # pdf for the report, png for the slides
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))


def subsample(df, start, end=None):
    """Rows of a month-indexed frame/series between start and end (inclusive)."""
    start = pd.Period(start, "M")
    end = pd.Period(end, "M") if end else df.index.max()
    return df[(df.index >= start) & (df.index <= end)]
