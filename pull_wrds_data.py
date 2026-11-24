# Downloads everything we need from WRDS into ./data
#
# usage:  python pull_wrds_data.py --user YOUR_USER --start 1925 --end 2025
#
# The daily file is pulled one year at a time (it is ~27 GB in full) and each
# year is saved as its own parquet file, so if the connection drops you can
# just rerun the same command and it skips the years that are already done.

import argparse
import os

import pandas as pd
import wrds

DATA_DIR = "data"

# CIZ (v2) version of the usual shrcd 10/11, exchcd 1/2/3 common stock filter
COMMON_STOCK_FILTER = """
    sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
    AND usincflg = 'Y' AND issuertype IN ('ACOR', 'CORP')
    AND primaryexch IN ('N', 'A', 'Q')
    AND conditionaltype = 'RW' AND tradingstatusflg = 'A'
"""


def pull_daily(db, start, end):
    out_dir = os.path.join(DATA_DIR, "crsp_daily")
    os.makedirs(out_dir, exist_ok=True)

    for year in range(start, end + 1):
        path = os.path.join(out_dir, f"{year}.parquet")
        if os.path.exists(path):
            print(f"daily {year}: already have it, skipping")
            continue

        sql = f"""
            SELECT permno, dlycaldt AS date, dlyret AS ret, dlyprc AS prc,
                   dlyhigh AS high, dlylow AS low, dlyvol AS vol
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31'
            AND {COMMON_STOCK_FILTER}
        """
        df = db.raw_sql(sql, date_cols=["date"])

        # smaller types so the files (and later the beta step) use less memory
        df["permno"] = df["permno"].astype("int32")
        for col in ["ret", "prc", "high", "low", "vol"]:
            df[col] = df[col].astype("float32")

        df.to_parquet(path, index=False)
        print(f"daily {year}: {len(df):,} rows")


def pull_monthly(db):
    sql = f"""
        SELECT permno, mthcaldt AS date, mthret AS ret, mthcap AS cap,
               mthprc AS prc, primaryexch
        FROM crsp.msf_v2
        WHERE {COMMON_STOCK_FILTER}
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df["permno"] = df["permno"].astype("int32")
    df.to_parquet(os.path.join(DATA_DIR, "crsp_monthly.parquet"), index=False)
    print(f"monthly CRSP: {len(df):,} rows, {df['date'].min()} to {df['date'].max()}")


def pull_factors(db):
    ff_daily = db.raw_sql("SELECT date, mktrf, smb, hml, rf FROM ff.factors_daily",
                          date_cols=["date"])
    ff_daily.to_parquet(os.path.join(DATA_DIR, "ff_daily.parquet"), index=False)
    print(f"FF daily: {len(ff_daily):,} rows")

    # FF3 + momentum, goes back to 1926 (the five factor table only starts in 1963)
    ff3 = db.raw_sql("SELECT date, mktrf, smb, hml, umd, rf FROM ff.factors_monthly",
                     date_cols=["date"])
    ff3.to_parquet(os.path.join(DATA_DIR, "ff3_monthly.parquet"), index=False)
    print(f"FF3 monthly: {len(ff3):,} rows")

    ff5 = db.raw_sql("SELECT date, mktrf, smb, hml, rmw, cma, umd, rf FROM ff.fivefactors_monthly",
                     date_cols=["date"])
    ff5.to_parquet(os.path.join(DATA_DIR, "ff5_monthly.parquet"), index=False)
    print(f"FF5 monthly: {len(ff5):,} rows")

    ps = db.raw_sql("SELECT date, ps_vwf FROM ff.liq_ps", date_cols=["date"])
    ps.to_parquet(os.path.join(DATA_DIR, "ps_liq_monthly.parquet"), index=False)
    print(f"PS liquidity: {len(ps):,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--start", type=int, default=1925)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    db = wrds.Connection(wrds_username=args.user)

    # small tables first so a dropped connection during the daily pull costs less
    pull_factors(db)
    pull_monthly(db)
    pull_daily(db, args.start, args.end)

    db.close()
    print("done")
