"""
Add sector and macro covariate features to the lag_60 datasets built by build_returns.py.

Writes, per test year Y (under data/sequences/ by default):
  lag_60_sector/returns_{Y}.parquet      lag_60 + 'sector' column                      (sector LSTM)
  lag_60_cov/returns_{Y}.parquet         stock + 5 covariate sequences of 60 lags      (covariate LSTM)
  lag_60_cov_sector/returns_{Y}.parquet  lag_60_cov + 'sector' column                  (covariate + sector LSTM)

Which outputs are built depends on which inputs you pass (--sectors and/or the 5 covariate files).
These inputs come from licensed sources and are NOT redistributed with this repo:
  --sectors    LSEG company sector CSV with columns 'Instrument' (RIC) and 'Economic_Sector'
               (the repo ships one: data/company_sector.csv)
  --oil        Bloomberg crude oil prices CSV, columns 'date', 'PX_LAST'
  --vix        Cboe VIX CSV, columns 'date', 'vix'
  --gold       LSEG gold futures (GCc1) CSV, columns 'Date', 'SETTLE'
  --rates      Federal Reserve daily interest rates CSV, columns 'date', 't10y3m', 'dgs10'

Covariates per day: oil return, VIX level, 10y-3m term spread, daily change of the 10y yield
('t10y_change'), gold return.
Each covariate is standardized with the mean / std of the 3 training years only.

Example (from the repository root):
  python data/add_features.py \
      --sectors data/company_sector.csv \
      --oil oil_prices.csv --vix vix.csv --gold gold_futures_GCc1.csv --rates Interest_Rate_Daily.csv
"""

import argparse
from pathlib import Path

import pandas as pd

LAGS = 60
TRAIN_YEARS = 3

DATA_DIR = Path(__file__).resolve().parent          # <repo>/data
SEQUENCES_DIR = DATA_DIR / 'sequences'              # output root of build_returns.py


def load_covariates(args):
    oil = pd.read_csv(args.oil, index_col='date', parse_dates=['date'])['PX_LAST']
    vix = pd.read_csv(args.vix, index_col='date', parse_dates=['date'])['vix']
    gold = pd.read_csv(args.gold, index_col='Date', parse_dates=['Date'])['SETTLE']
    rates = pd.read_csv(args.rates, index_col='date', parse_dates=['date']).sort_index()
    return {
        'oil_return': oil.pct_change(fill_method=None).dropna(),
        'vix': vix,
        't10y3m': rates['t10y3m'].dropna(),
        't10y_change': rates['dgs10'].dropna().diff().round(2),
        'gold_return': gold.pct_change(fill_method=None).dropna(),
    }


def merge_covariate(df, covariate, name, year, dates):
    """Attach a '{name}_sequence' column holding the LAGS standardized covariate values before each date."""
    cov = covariate.loc[f'{year - TRAIN_YEARS - 1}-11-01':f'{year}-12-31']
    if cov.isna().any():
        print(f'  {name}: {cov.isna().sum()} missing values interpolated')
        cov = cov.interpolate()
    missing = [d for d in dates if d not in cov.index]
    if missing:
        print(f'  {name}: {len(missing)} trading days not in covariate data, interpolated')
        cov = cov.reindex(cov.index.union(pd.DatetimeIndex(missing))).sort_index().interpolate()

    train = covariate.loc[f'{year - TRAIN_YEARS}-01-01':f'{year - 1}-12-31']
    cov = (cov - train.mean()) / train.std()

    seqs = [cov.iloc[i - LAGS:i].values for i in range(LAGS, len(cov))]
    seqs = pd.DataFrame({'date': cov.index[LAGS:], f'{name}_sequence': seqs})
    return df.merge(seqs, on='date', how='left')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default=str(SEQUENCES_DIR),
                   help='Output root of build_returns.py (contains lag_60/). Default: data/sequences')
    p.add_argument('--sectors')
    p.add_argument('--oil')
    p.add_argument('--vix')
    p.add_argument('--gold')
    p.add_argument('--rates')
    p.add_argument('--start_year', type=int, default=1998)
    p.add_argument('--end_year', type=int, default=2024)
    args = p.parse_args()

    cov_files = [args.oil, args.vix, args.gold, args.rates]
    use_cov = all(cov_files)
    if any(cov_files) and not use_cov:
        p.error('covariates need all of --oil, --vix, --gold, --rates')
    use_sector = args.sectors is not None
    if not (use_cov or use_sector):
        p.error('pass --sectors and/or the covariate files')

    root = Path(args.data_dir)
    outputs = (['lag_60_sector'] if use_sector else []) + (['lag_60_cov'] if use_cov else []) \
        + (['lag_60_cov_sector'] if use_cov and use_sector else [])
    for sub in outputs:
        (root / sub).mkdir(parents=True, exist_ok=True)

    if use_sector:
        sectors = pd.read_csv(args.sectors, usecols=['Instrument', 'Economic_Sector'])
        stock_to_sector = dict(zip(sectors['Instrument'], sectors['Economic_Sector']))
    if use_cov:
        covariates = load_covariates(args)

    for year in range(args.start_year, args.end_year + 1):
        print(f'Test year {year}')
        base = pd.read_parquet(root / 'lag_60' / f'returns_{year}.parquet')

        if use_sector:
            base.assign(sector=base['stock'].map(stock_to_sector)) \
                .to_parquet(root / 'lag_60_sector' / f'returns_{year}.parquet')

        if use_cov:
            dates = base['date'].unique().tolist()
            cov_df = base.rename(columns={'sequence': 'stock_return_sequence'})
            for name, series in covariates.items():
                cov_df = merge_covariate(cov_df, series, name, year, dates)
            n_nan = cov_df.isnull().any(axis=1).sum()
            if n_nan:
                print(f'  WARNING: {n_nan} rows with missing covariates')
            cov_df.to_parquet(root / 'lag_60_cov' / f'returns_{year}.parquet')

            if use_sector:
                cov_df.assign(sector=cov_df['stock'].map(stock_to_sector)) \
                    .to_parquet(root / 'lag_60_cov_sector' / f'returns_{year}.parquet')


if __name__ == '__main__':
    main()
