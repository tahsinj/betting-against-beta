# Replication notes

What we took from each paper, how we implemented it, what we left out, and how our
numbers compare to the published ones. The README covers how to run the code; this
file is for checking the finance side.

Papers:

- **FP** = Frazzini and Pedersen (2014), "Betting against beta", JFE 111(1).
- **NV** = Novy-Marx and Velikov (2022), "Betting against betting against beta", JFE 143(1).
  We compare against the published version. An earlier 2018 working paper used a
  sample ending in 2017 and has slightly different numbers.

All returns and alphas below are in % per month. t-stats are in brackets.

## 1. Sample periods

| Part | Period | Months | Why |
|---|---|---|---|
| FP replication (step 4) | Jan 1931 - Mar 2012 | 975 | FP's US sample ends March 2012. Betas need about 3 years of daily data first, so our first portfolios are in the early 1930s. |
| FP five-factor alphas | Jan 1968 - Mar 2012 | 531 | The Pastor-Stambaugh liquidity factor starts in 1968. |
| NV tests (step 5) | Jan 1968 - Dec 2019 | 624 | NV's published sample. FF5 factors need to start after July 1963. |
| Gibbs cost check (step 5) | Jan 1968 - Dec 2009 | 504 | Hasbrouck's estimates stop in 2009. |
| Post-publication (step 6) | Apr 2012 - Dec 2025 | 165 | Everything after FP's sample. This is our own extension, not from either paper. |

The dates are set in `helpers.py` (`FP_START`, `FP_END`, `NV_START`, `NV_END`, `POST_START`, `POST_END`).

## 2. Data

- CRSP daily and monthly stock files (new CIZ format) from WRDS, US common stocks on
  NYSE, AMEX and NASDAQ (`pull_wrds_data.py`). Monthly returns include delisting returns.
- Fama-French 3 factors and momentum (UMD), Fama-French 5 factors, and the
  Pastor-Stambaugh traded liquidity factor, all from WRDS.
- AQR's maintained US BAB factor (AQR data library), only used to check our BAB.
- Hasbrouck's Gibbs effective cost estimates, only used for the cost robustness check.

## 3. Frazzini and Pedersen (2014)

### Implemented

| FP | What it is | Our code |
|---|---|---|
| Sec 3.1, eq. 14 | Beta = correlation x (stock vol / market vol). Vol: 1 year of daily log returns (at least 120 days). Correlation: 5 years of overlapping 3-day log returns (at least 750 days). | `step1_betas.py` |
| Sec 3.1, eq. 15 | Shrink betas toward the cross-sectional mean, which FP set to 1: 0.6 x estimate + 0.4 x 1 | `step1_betas.py` |
| Sec 3.2, eq. 16-17 | BAB: rank stocks on beta, weight by distance of the rank from the average rank, scale the low-beta leg up and the high-beta leg down to a beta of 1, long low minus short high. Rebalanced monthly. | `step3_portfolios.py` |
| Table 3 | 10 beta deciles (NYSE breakpoints, equal weighted) plus BAB: returns, alphas (CAPM, FF3, Carhart, + liquidity), ex ante and realized betas, vol, Sharpe | `step4_replication.py`, `table1_deciles_1931_2012` |
| Table 3 (BAB column), Table B1 | BAB factor regressions | `table2_bab_regressions_1931_2012` |
| AQR data | Correlation of our BAB with AQR's BAB | `table3_aqr_comparison`, `fig2_bab_vs_aqr` |

Extra things we report that FP don't: value weighted deciles (`table1b_deciles_vw_1931_2012`)
and the long and short legs of BAB separately (`table2b_bab_legs_1931_2012`).

### FP Table 3 vs ours

| | FP | Ours |
|---|---|---|
| BAB excess return | 0.70 [7.12] | 0.77 [7.23] |
| BAB CAPM alpha | 0.73 [7.44] | 0.81 [5.87] |
| BAB FF3 alpha | 0.73 [7.39] | 0.81 [6.14] |
| BAB Carhart alpha | 0.55 [5.59] | 0.60 [4.69] |
| BAB five-factor alpha | 0.55 [4.09] | 0.62 [3.30] |
| BAB realized beta | -0.06 | -0.06 |
| BAB volatility (%/yr) | 10.75 | 11.51 |
| BAB Sharpe ratio | 0.78 | 0.80 |
| P1 (low beta) excess return | 0.91 | 1.01 |
| P10 (high beta) excess return | 0.97 | 1.14 |
| P1 / P10 CAPM alpha | 0.52 / -0.10 | 0.58 / -0.08 |
| P1 / P10 ex ante beta | 0.64 / 1.70 | 0.61 / 1.77 |
| P1 / P10 realized beta | 0.67 / 1.85 | 0.65 / 1.89 |
| P1 / P10 Sharpe ratio | 0.70 / 0.28 | 0.79 / 0.32 |
| Correlation with AQR's BAB, 1931-2012 | | 0.971 |

The pattern matches FP: flat average returns across deciles, alphas falling from
low to high beta, Sharpe ratios falling, BAB with a large alpha and near zero beta.
Our decile returns are about 0.1% per month higher across the board. A likely reason
is that our sample starts in 1931 while FP's starts as soon as betas are available,
so FP include more of the 1929-1931 crash. Our t-stats are a bit lower because we use
Newey-West standard errors (6 lags).

### Not implemented

- International equities (FP Table 4, 20 countries) and all other asset classes:
  Treasuries (Table 6), corporate bonds / credit (Table 7), equity index, country bond,
  currency and commodity futures (Table 8). We only have US stock data.
- The funding liquidity tests: BAB returns vs the TED spread (Table 9), beta
  compression (Table 10) and BAB's conditional market beta (Table 11).
- Robustness tables in FP's appendix (other beta windows, size and period splits,
  idiosyncratic volatility controls, FP's own "value weighted" BAB in online appendix Table B9).
- The theory (Propositions 1-4). This goes in the report, not the code.

## 4. Novy-Marx and Velikov (2022)

### Implemented

| Our test | NV section / table | What it checks | Output |
|---|---|---|---|
| Test 1 | Sec 2.1, Fig 2, Table C.8 | FP's rank weighting vs simple sorts: equal and value weighted bottom/top thirds by beta, and NV's rank x market cap weights | `nv_test1_weighting` |
| Test 2 | Sec 3, Table 1, footnote 13 | How many BAB dollars go into nano caps (smallest stocks making up 1% of market cap) and into each NYSE size decile | `nv_test2_small_stock_exposure`, `nv_test2b_holdings_by_size_decile`, `fig3_nano_caps` |
| Test 3a | Sec 4 (and FP's ex ante vs realized beta) | Ex ante vs realized betas of BAB's two legs | `nv_test3a_leg_betas` |
| Test 3b | Appendix D (different version, see below) | BAB built from plain OLS betas instead of FP betas | `nv_test3b_ols_beta_bab` |
| Test 3c | Sec 4.1 (different method, see below) | Do FP betas forecast future betas better than OLS betas? | `nv_test3c_beta_forecasts` |
| Test 3d | Sec 4.3, Fig 8 | The market's value weighted FP-beta (should be exactly 1) and how it moves with log(1y market vol / 5y market vol) | `nv_test3d_market_fp_beta`, `fig6_market_fp_beta` |
| Test 3e | Sec 5, Table 5 specs 1-3 | BAB regressed on current and lagged market returns, and market x vol ratio | `nv_test3e_bab_lagged_beta` |
| Test 4 | Sec 3.1, Table 2 (total only), Fig 4, Table C.10 | Turnover, trading costs, net returns, net FF5 alpha | `nv_test4_transaction_costs`, `fig4_net_of_costs` |
| Test 4b | (our addition) | Same costs with Hasbrouck's Gibbs estimates, 1968-2009 | `nv_test4b_costs_gibbs_1968_2009` |
| Test 5 | Sec 2.1, Tables C.7 and C.8 | Is BAB explained by profitability (RMW) and investment (CMA)? FF3 / FF5 / FF6 regressions | `nv_test5_bab_ff6`, `nv_test5b_value_bab_ff6`, `nv_test5c_net_bab_ff6` |

### NV numbers vs ours (1968-2019)

| | NV | Ours | Our table |
|---|---|---|---|
| BAB mean | 1.05 [7.69] | 1.03 [7.60] | test1 |
| BAB Sharpe ratio | 1.07 | 1.05 | test1 |
| Equal weighted thirds: mean | 1.02 [7.53] | 1.00 [7.46] | test1 |
| Correlation of BAB with equal weighted thirds | 99.6% | 99.7% | test1 |
| Value weighted thirds: mean | 0.55 [3.47] | 0.55 [3.51] | test1 |
| Value weighted thirds: Sharpe | 0.48 | 0.49 | test1 |
| Value weighted thirds: FF5 / FF6 alpha | 0.24 [1.68] / 0.14 [0.96] | 0.25 [1.65] / 0.15 [0.90] | test5b |
| Value weighted thirds: RMW / CMA loading | 0.44 / 0.55 | 0.44 / 0.55 | test5b |
| BAB FF3 alpha | 0.94 [7.05] | 0.91 [5.51] | test5 |
| BAB FF5 alpha | 0.65 [5.11] | 0.63 [3.91] | test5 |
| BAB FF6 alpha | 0.51 [4.13] | 0.48 [2.96] | test5 |
| BAB RMW / CMA loading (FF5) | 0.56 / 0.39 | 0.57 / 0.42 | test5 |
| BAB dollars in nano caps | $1.05 | $1.05 | test2 |
| Smallest NYSE decile, long / short (cents per $1) | 94.6 / 26.1 | 94.4 / 26.0 | test2b |
| Total long / short (cents per $1) | 156.4 / 72.5 | 156.3 / 72.5 | test2b |
| Market FP-beta, mean / std | 1.046 / 0.094 | 1.046 / 0.094 | test3d |
| Slope of market FP-beta on log vol ratio | -0.33 [-4.77], R2 48% | -0.18 [-4.23], R2 36% | test3d |
| BAB market beta (Table 5 spec 1) | -0.06 [-1.93] | -0.06 [-0.97] | test3e |
| BAB on market lag 1 / lag 2 | 0.16 / 0.12 | 0.16 / 0.11 | test3e |
| Market x vol ratio (spec 3) | -0.13 [-4.61] | -0.10 [-1.99] | test3e |
| Turnover (%/yr, avg of buys and sells) | 214.4 | 210.4 | test4 |
| Trading cost: BAB | 0.58 | 0.40 | test4 |
| Trading cost: BAB without smallest NYSE decile | 0.22 | 0.16 | test4 |
| Trading cost: rank x cap BAB | 0.08 | 0.06 | test4 |
| Net mean: BAB | 0.47 [3.35] | 0.63 [4.60] | test4 |
| Net mean: rank x cap BAB | 0.42 [2.82] | 0.45 [2.98] | test4 |
| Net FF5 alpha: BAB | 0.17 [1.32] | 0.24 [1.37] | test4 |

Trading costs, 1968-2009 only (test 4b). NV don't report this period directly, so the
NV column is backed out from their full sample and 2010-2019 numbers:

| | Abdi-Ranaldo (ours) | Gibbs (ours) | Implied by NV |
|---|---|---|---|
| BAB | 0.45 | 0.68 | about 0.68 |
| BAB without smallest NYSE decile | 0.18 | 0.24 | about 0.25 |
| rank x cap BAB | 0.07 | 0.08 | about 0.09 |

### Differences from NV worth knowing

- **Spread measure.** Our main cost numbers use Abdi and Ranaldo (2017) spreads from
  daily high/low/close prices. NV's method (Novy-Marx and Velikov 2016) is built on
  Hasbrouck's Gibbs estimates. With Gibbs estimates our costs line up with NV (table above),
  so the gap in test 4 is down to the spread measure.
- **Net alpha.** NV's net alpha is a "generalized alpha" that also charges trading
  costs on the factors. Ours is a plain regression of net BAB returns on the normal
  (gross) factors. Same conclusion (not significant), but not the same number.
- **Test 3b.** NV's Appendix D uses 5-year OLS betas on 3-day returns. Ours uses 1-year
  OLS betas on daily returns (shrunk the same way as FP). Ours is a harsher test of
  non-synchronous trading: the daily OLS betas of small stocks are too low, so BAB
  built on them ends up with a realized beta of 0.28 instead of about zero.
- **Test 3c.** NV (Sec 4.1, Fig 5) sort stocks on each beta estimator and compare the
  realized beta spread between high and low portfolios. We instead check stock by
  stock how well each estimate forecasts the beta over the next 12 months (bias,
  RMSE and slope). Both ask "is the FP estimator better?", but the numbers can't be
  compared directly.
- **Test 3d slope.** We match the mean and std of the market FP-beta exactly, but get a
  smaller slope on the vol ratio. NV's 2018 draft reported -0.21, so they changed
  something between versions that we can't see. We tried a couple of other vol ratio
  definitions without closing the gap.
- **t-stats.** Ours are lower almost everywhere because we use Newey-West standard
  errors with 6 lags (60 lags for test 3d, as NV do). NV don't say which standard errors
  they use for most tables.

### Not implemented

- Directly hedged BAB (Sec 2.2, Fig 3, Table C.9): hedging with the equal or value
  weighted market instead of leverage.
- Simple beta arbitrage (Table 5 specs 4-6, Fig 9, Table C.7 specs 6-10).
- Hedging bias regressions on the market, the BAB legs and the 25 size/value portfolios (Table 3).
- Volatility and correlation facts (Fig 7) and the other beta estimators: Dimson,
  Vasicek, Welch (Fig 5, Fig 6, Appendix E).
- TED spread beta compression test (Table 4, Appendix F).
- Turnover by size decile (Table 2). We only report total turnover.
- Rank vs equal weighting for value and momentum (Appendix B).

## 5. Our extension: after FP's sample (step 6)

Not from either paper. We split 1968 - Mar 2012 and Apr 2012 - Dec 2025 and compare
BAB, BAB net of costs, value weighted thirds, rank x cap BAB (gross and net) and AQR's
BAB (`post_table1_before_after`, `post_table2_change_in_mean`, `fig5_post_publication`,
decile tables `post_table3_deciles_ew_2012_2025` and `_vw_`).

| Apr 2012 - Dec 2025 | Mean | FF6 alpha |
|---|---|---|
| BAB (ours) | 0.72 [3.49] | 0.61 [3.43] |
| BAB net of costs | 0.48 [2.28] | 0.39 [2.10] |
| Value weighted thirds | -0.04 [-0.11] | -0.22 [-0.84] |
| rank x cap BAB | -0.02 [-0.05] | -0.23 [-0.93] |
| AQR BAB | 0.52 [2.74] | 0.37 [2.11] |

The equal weighted version kept working after 2012, but the versions an investor
could realistically trade at size (value weighted, rank x cap) earned nothing. The
drop in the mean after 2012 is not statistically significant for any version
(`post_table2_change_in_mean`), partly because 165 months is a short sample.

## 6. Things for the group to check

- The report should state FP's assumptions and our choices that differ from the papers
  (section 4 "Differences from NV", Newey-West t-stats, the 1931 start).
- FP Table 3 returns are about 0.1% higher in our version for every decile. Is the
  start date explanation good enough, or should we try starting earlier?
- Which NV numbers to put in the report tables: the published 2022 ones above, not the 2018 draft.
- The report should say clearly which FP and NV results we did not replicate (lists above)
  and why (data access, scope).
