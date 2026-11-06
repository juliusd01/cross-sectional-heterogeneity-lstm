"""
Build survivorship-bias-free S&P 500 return datasets from raw LSEG exports.

Pipeline (one study period per test year Y):
  1. Replay the S&P 500 joiner/leaver list to get the index constituents at the
     end of December Y-1 (the last month of the training period).
  2. Take the daily prices of exactly these constituents for Y-3-01-01 .. Y-12-31
     (3 training years + 1 test year). Stocks that leave the index later stay in
     the test year as long as prices exist -> no survivorship bias.
  3. Compute simple daily returns and standardize them with the mean / std of
     the 3 training years only (no look-ahead).
  4. Label each (stock, day) with Class1 = 1 if its return is >= the
     cross-sectional median of that day, else Class0 = 1.
  5. Write two feature sets per year (under data/sequences/ by default):
       lag_60/returns_{Y}.parquet   -> 60 standardized lagged returns ('sequence'), for the LSTMs
       tabular/returns_{Y}.parquet  -> last-day + cumulative raw returns, for the random forest
     With --save_study_periods the per-period price panels are also written to
     data/prices/test_{Y}.csv, where the analysis scripts expect them.

Required raw inputs (download yourself, e.g. from LSEG Workspace):
  --prices  CSV with a 'Date' column and one column of daily prices per LSEG RIC
            (all current and former S&P 500 members, incl. delisted RICs like 'AAL.N^B97').
  --movers  Excel file with the S&P 500 joiners/leavers history, no header,
            columns: Status ('Joiner'/'Leaver'), Issuer, Code (RIC), Date.

Example (from the repository root):
  python data/build_returns.py \
      --prices  data/Daily_Prices.csv \
      --movers  "data/SPX_index leavers & joiners_11-Nov-2025.xlsx"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = 60
CUM_HORIZONS = list(range(2, 21)) + [40, 60]
TRAIN_YEARS = 3

DATA_DIR = Path(__file__).resolve().parent          # <repo>/data
SEQUENCES_DIR = DATA_DIR / 'sequences'              # read by src/train.py via --input_subdir
PRICES_DIR = DATA_DIR / 'prices'                    # test_{Y}.csv, read by the analysis scripts


# --------------------------------------------------------------------------- #
# 1. Index membership
# --------------------------------------------------------------------------- #
def load_movers(path):
    movers = pd.read_excel(path, header=None)
    movers.columns = ['Status', 'Issuer', 'Code', 'Date']
    movers['Date'] = pd.to_datetime(movers['Date'])
    movers['Code'] = movers['Code'].str.replace('(expired)', '', regex=False).str.strip()
    return movers.sort_values(by='Date')


def constituents_at(movers, cutoff):
    """Index members after replaying all joiner/leaver events up to and including `cutoff`."""
    members = set()
    for _, row in movers[movers['Date'] <= cutoff].iterrows():
        if row['Status'] == 'Joiner':
            members.add(row['Code'])
        elif row['Status'] == 'Leaver':
            members.discard(row['Code'])
    return sorted(members)


# --------------------------------------------------------------------------- #
# 2. Study period prices
# --------------------------------------------------------------------------- #
def study_period_prices(prices, movers, test_year):
    cutoff = pd.Timestamp(f'{test_year - 1}-12-31')
    stocks = constituents_at(movers, cutoff)
    missing = [s for s in stocks if s not in prices.columns]
    if missing:
        print(f'  {len(missing)} constituents have no price data and are skipped: {missing}')
    stocks = [s for s in stocks if s in prices.columns]
    period = prices.loc[f'{test_year - TRAIN_YEARS}-01-01':f'{test_year}-12-31', stocks]
    return period.dropna(axis=1, how='all')


# --------------------------------------------------------------------------- #
# 3.-5. Returns, standardization, labels, features
# --------------------------------------------------------------------------- #
def build_year(prices, movers, test_year):
    period_prices = study_period_prices(prices, movers, test_year)

    raw_returns = period_prices.pct_change(fill_method=None).iloc[1:]

    # standardize with statistics of the training years only
    train = raw_returns.loc[f'{test_year - TRAIN_YEARS}-01-01':f'{test_year - 1}-12-31']
    mean_return = train.mean().mean()
    mean_volatility = train.std().mean()
    std_returns = (raw_returns - mean_return) / mean_volatility

    # cross-sectional median per day (over all stocks with a return that day)
    daily_median = std_returns.median(axis=1)

    lag_rows, tab_rows = [], []
    for stock in std_returns.columns:
        valid = std_returns[stock].notna()
        dates = std_returns.index[valid]
        vals = std_returns[stock].values[valid.values]
        vals_raw = raw_returns[stock].values[valid.values]
        n = len(vals)
        if n <= LAGS:
            continue
        idx = np.arange(LAGS, n)
        target_dates = dates[LAGS:]

        windows = np.lib.stride_tricks.sliding_window_view(vals, LAGS)[: n - LAGS]
        base = {'stock': stock, 'date': target_dates, 'return': vals[LAGS:]}

        lag_rows.append(pd.DataFrame({**base, 'sequence': [w.copy() for w in windows]}))

        tab = {**base, 'return_t1': vals_raw[idx - 1]}
        cum = np.ones(len(idx))
        for t in range(1, max(CUM_HORIZONS) + 1):
            cum = cum * (1 + vals_raw[idx - t])
            if t in CUM_HORIZONS:
                tab[f'cum_return_t{t}'] = cum - 1
        tab_rows.append(pd.DataFrame(tab))

    labels = lambda df: df.assign(
        Class0=(df['return'] < df['date'].map(daily_median)).astype('int64'),
        Class1=(df['return'] >= df['date'].map(daily_median)).astype('int64'),
    )
    lag_df = labels(pd.concat(lag_rows, ignore_index=True))
    tab_df = labels(pd.concat(tab_rows, ignore_index=True))
    return period_prices, lag_df, tab_df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prices', required=True, help='Daily price CSV (Date + one column per RIC)')
    p.add_argument('--movers', required=True, help='S&P 500 joiners/leavers Excel file')
    p.add_argument('--out_dir', default=str(SEQUENCES_DIR),
                   help='Output root for lag_60/ and tabular/; src/train.py reads '
                        '<out_dir>/<input_subdir>. Default: data/sequences')
    p.add_argument('--start_year', type=int, default=1998, help='First test year (default 1998)')
    p.add_argument('--end_year', type=int, default=2024, help='Last test year (default 2024)')
    p.add_argument('--save_study_periods', action='store_true',
                   help='Also write the per-period price panels to <study_periods_dir>/test_{Y}.csv')
    p.add_argument('--study_periods_dir', default=str(PRICES_DIR),
                   help='Where --save_study_periods writes the panels. Default: data/prices')
    args = p.parse_args()

    out = Path(args.out_dir)
    for sub in ['lag_60', 'tabular']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    study_dir = Path(args.study_periods_dir)
    if args.save_study_periods:
        study_dir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(args.prices, parse_dates=['Date'], index_col='Date').sort_index()
    prices = prices.loc['1995-01-01':]
    movers = load_movers(args.movers)

    for year in range(args.start_year, args.end_year + 1):
        print(f'Test year {year}')
        period_prices, lag_df, tab_df = build_year(prices, movers, year)
        if args.save_study_periods:
            period_prices.to_csv(study_dir / f'test_{year}.csv')
        lag_df.to_parquet(out / 'lag_60' / f'returns_{year}.parquet', engine='pyarrow', index=False)
        tab_df.to_parquet(out / 'tabular' / f'returns_{year}.parquet', engine='pyarrow', index=False)
        print(f'  {period_prices.shape[1]} stocks, {len(lag_df)} samples written')


if __name__ == '__main__':
    main()
