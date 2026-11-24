# ACTSC 970 Project: Betting Against Beta

Replication of Frazzini and Pedersen (2014) on US stocks and the tests from
Novy-Marx and Velikov (2022), plus an out-of-sample check for 2013-2025.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

CRSP (CIZ / v2 format) and the Fama-French and Pastor-Stambaugh factors come
from WRDS. The first run asks for your WRDS password and a Duo push:

```bash
python pull_wrds_data.py --user YOUR_WRDS_USERNAME --start 1925 --end 2025
```

The AQR BAB factor (`data/aqr_bab_monthly.xlsx`) is the "Betting Against Beta:
Equity Factors, Monthly" file from the AQR data library.

Hasbrouck's Gibbs effective cost estimates (`data/crspGibbs2009v01.txt`, used only as a
robustness check on trading costs) are the text file on his
[updated Gibbs estimates page](https://pages.stern.nyu.edu/~jhasbrou/Research/GibbsCurrent/gibbsCurrentIndex.html).
They run up to 2009.

None of the data is included in this repo (CRSP is licensed through WRDS). All scripts
read from and write to a local `data/` folder.

## Papers

- Frazzini, A. and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics* 111(1), 1-25.
- Novy-Marx, R. and Velikov, M. (2022). Betting against betting against beta. *Journal of Financial Economics* 143(1), 80-106.

## Running

Run the steps in order. Each one reads the output of the previous steps from `data/`.

| Script | What it does | Output |
|---|---|---|
| `step1_betas.py` | FP betas (1y vol, 5y 3-day correlation, shrinkage) and 1y OLS betas at every month end | `data/betas.parquet` |
| `step2_spreads.py` | Abdi-Ranaldo (2017) monthly spread estimates from daily high/low/close, plus Hasbrouck's Gibbs costs | `data/spreads.parquet`, `data/gibbs_costs.parquet` |
| `step3_portfolios.py` | BAB factor and variants, beta deciles (equal and value weighted), small-stock exposure, turnover and trading costs | `data/bab_series.parquet`, `data/deciles.parquet`, `data/bab_holdings.parquet` |
| `step4_replication.py` | FP tables for 1931 to March 2012 and comparison with AQR's BAB | tables 1-3, figures 1-2 |
| `step5_novy_marx_velikov.py` | NV tests on their sample (1968-2019) | `nv_test*` tables, figures 3, 4, 6 |
| `step6_post_publication.py` | Before vs after FP's sample (April 2012 to 2025) | `post_*` tables, figure 5 |

`helpers.py` has the sample dates, factor models and the regression / table functions.
Tables are saved as `.csv` and `.tex` in `output/tables/`, figures as `.pdf` and `.png` in `output/figures/`.

Step 1 is the slow one (loads all of the daily file, roughly 1-2 GB of memory).

## Choices worth knowing about

- Market return for betas is the FF market (`mktrf + rf`), which is the CRSP value-weighted index.
- Betas use daily log excess returns. The OLS beta is shrunk with the same weight (0.6) as the FP beta, so the two versions of BAB only differ in the estimator.
- Portfolios are formed at the end of month t and held over t+1. Monthly CRSP returns already include delisting returns.
- Beta deciles use NYSE breakpoints. FP's Table 3 equal weights them; we also report value weighted deciles.
- NV's equal and value weighted BAB use the bottom and top thirds of stocks by beta. Their "implementable" BAB weights by rank times market cap.
- Nano caps follow NV: the smallest stocks that together make up 1% of total market cap. Micro caps are below the NYSE 20th percentile.
- Turnover in the NV tables is the average of purchases and sales per year, as in NV.
- Trading cost = dollars traded x half the estimated spread. Missing spreads are filled with the median of stocks in the same size decile that month.
- Abdi-Ranaldo spreads give lower costs than NV report. NV's method is built on Hasbrouck's Gibbs estimates, so `nv_test4b` redoes the costs with those for 1968-2009 (the years they cover). Gibbs costs for BAB come out about 50% higher, close to what NV's numbers imply for the same years.
- Alphas use Newey-West standard errors with 6 lags. FP5 = Carhart + Pastor-Stambaugh traded liquidity factor (starts 1968). FF5/FF6 start in July 1963.
