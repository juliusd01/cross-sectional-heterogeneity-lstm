"""Turn the raw price panels into the lag-60 sequence files the models train on.

One study period = four calendar years: three of training/validation and one
out-of-sample test year.  `data/prices/test_{year}.csv` holds the daily price
panel for that whole four-year block, with the columns being the S&P 500
constituents as of the end of the training window.

For each stock and each trading day t the script writes
  - `sequence`        the 60 standardised returns ending at t-1
  - `return`          the return realised on t (the target day), in percent
  - `Class0`/`Class1` whether that return was below / above the cross-sectional
                      median of day t
  - `sector`          the TRBC economic sector of the stock
and, for the covariate variants, one `<name>_sequence` column per macro series.

Standardisation
---------------
The *model inputs* are standardised with a single mean and standard deviation
computed on the training window only, pooled over all stocks and days, as
described in section 2.2 of the paper.

The `return` column is deliberately NOT standardised: it is the realised return
in percent, and it is what every performance figure is computed from.  The
original run left it in standardised units, which understated all reported
returns by a factor of sigma; see README, "Return units".

`--standardisation reference` instead uses the constants in
`data/standardisation_reference.csv`, which were recovered from the published
prediction files.  Use it when you want rebuilt sequences to sit on exactly the
same scale as the original run.

NOTE: the script that produced the original sequence files was not preserved.
This is a reimplementation from the paper's description; it reproduces the
documented method but not necessarily the original universe filter byte for
byte.  Compare against `data/standardisation_reference.csv` after a rebuild.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .paths import COMPANY_SECTOR, COVARIATES, DATA, PRICES, SEQUENCES, STUDY_YEARS

NUM_LAGS = 60

# Macro-financial covariates, in the order they are fed to the LSTM.  The order
# matters: the training script picks up every column ending in `sequence`, so it
# determines the input dimension order of a saved covariate model.
COVARIATE_COLUMNS = ['oil_return', 'gold_return', 'vix', 't10y3m', 't10y_change']

def _variant(covariates=False, sector=False, tabular=False) -> dict:
    return dict(covariates=covariates, sector=sector, tabular=tabular)


VARIANTS = {
    'lag_60': _variant(),
    'lag_60_sector': _variant(sector=True),
    'lag_60_cov': _variant(covariates=True),
    'lag_60_cov_sector': _variant(covariates=True, sector=True),
    # flat features for the tree benchmarks, which have no memory
    'tabular': _variant(tabular=True),
}

# Cumulative-return horizons used by the random forest and the gradient
# boosting benchmark: m in {1..20} union {40, 60}, as in Fischer and Krauss (2018).
TABULAR_HORIZONS = list(range(1, 21)) + [40, 60]


def load_prices(year: int) -> pd.DataFrame:
    path = PRICES / f'test_{year}.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found.  The Refinitiv price panels are not redistributable;\n'
            'see data/README.md for the required format.'
        )
    prices = pd.read_csv(path, parse_dates=['Date'])
    return prices.sort_values('Date').reset_index(drop=True)


def daily_returns(prices: pd.DataFrame, bridge_gaps: bool = False) -> pd.DataFrame:
    """Simple one-day returns in percent, long format, missing values dropped.

    With `bridge_gaps`, isolated interior holes in the price series are linearly
    interpolated first.  A single missing print would otherwise invalidate the
    sixty windows that span it, and the published run kept those windows.
    """
    px = prices.drop(columns='Date')
    if bridge_gaps:
        px = px.interpolate(method='linear', limit_area='inside')
    returns = px.pct_change(fill_method=None) * 100
    returns.insert(0, 'Date', prices['Date'])
    returns = returns.iloc[1:]
    long = returns.melt(id_vars='Date', var_name='stock', value_name='return')
    return long.dropna(subset=['return']).rename(columns={'Date': 'date'})


def sector_map() -> dict:
    sectors = pd.read_csv(COMPANY_SECTOR, usecols=['Instrument', 'Economic_Sector'])
    return dict(zip(sectors['Instrument'], sectors['Economic_Sector']))


def standardisation(year: int, train: pd.Series, mode: str) -> tuple[float, float]:
    if mode == 'reference':
        ref = pd.read_csv(DATA / 'standardisation_reference.csv').set_index('study_period')
        if year not in ref.index:
            raise SystemExit(f'no reference constants for study period {year}')
        row = ref.loc[year]
        return float(row['mu_train_pct']), float(row['sigma_train_pct'])
    return float(train.mean()), float(train.std())


def load_covariates(dates: pd.Series) -> pd.DataFrame:
    if not COVARIATES.exists():
        raise SystemExit(
            f'{COVARIATES} not found.  The covariate variants need the five macro\n'
            'series described in data/README.md.'
        )
    cov = pd.read_csv(COVARIATES, parse_dates=['date']).sort_values('date')
    missing = [c for c in COVARIATE_COLUMNS if c not in cov.columns]
    if missing:
        raise SystemExit(f'{COVARIATES} is missing columns: {", ".join(missing)}')
    cov = cov.set_index('date').reindex(pd.DatetimeIndex(sorted(set(dates))))
    return cov[COVARIATE_COLUMNS].interpolate(method='linear')


def build_period(year: int, variant: str, mode: str) -> pd.DataFrame:
    """Build the sequence table for one study period."""
    opts = VARIANTS[variant]
    prices = load_prices(year)
    # `long` carries only genuinely observed returns: it fixes the labels, the
    # cross-sectional median and which days are admissible as a target.
    # `filled` bridges isolated gaps and supplies the lag windows.
    long = daily_returns(prices)
    filled = daily_returns(prices, bridge_gaps=True)

    is_test = long['date'].dt.year == year
    mu, sigma = standardisation(year, long.loc[~is_test, 'return'], mode)
    long['z'] = (long['return'] - mu) / sigma
    filled['z'] = (filled['return'] - mu) / sigma

    # Cross-sectional median split of the target day's return.  The split is
    # rank based, so standardising first does not change the labels.
    median = long.groupby('date')['return'].transform('median')
    long['Class1'] = (long['return'] >= median).astype(int)
    long['Class0'] = 1 - long['Class1']

    wide = filled.pivot(index='date', columns='stock', values='z').sort_index()
    if opts['covariates']:
        cov = load_covariates(wide.index)
        cov_train = cov[cov.index.year != year]
        cov_z = (cov - cov_train.mean()) / cov_train.std()

    rows = []
    stoi = sector_map() if opts['sector'] else {}
    labels = long.set_index(['date', 'stock'])[['Class0', 'Class1', 'return']]

    for stock in wide.columns:
        series = wide[stock]
        # A window needs all 60 lags present in the gap-bridged series; the
        # target day additionally has to be genuinely observed, which the
        # `labels` lookup below enforces.  Delisted names simply stop
        # contributing windows.
        values = series.to_numpy()
        valid = ~np.isnan(values)
        for i in range(NUM_LAGS, len(series)):
            if not valid[i - NUM_LAGS:i].all():
                continue
            date = series.index[i]
            if date.year != year and date.year < year - 3:
                continue
            key = (date, stock)
            if key not in labels.index:
                continue
            window = values[i - NUM_LAGS:i]
            row = {
                'stock': stock,
                'date': date,
                # realised return in percent -- the sequence stays standardised
                'return': float(labels.at[key, 'return']),
                'Class0': int(labels.at[key, 'Class0']),
                'Class1': int(labels.at[key, 'Class1']),
            }
            if opts['tabular']:
                # cumulative return over the last m days, ending at t-1
                row['return_t1'] = window[-1]
                for m in TABULAR_HORIZONS[1:]:
                    row[f'cum_return_{m}'] = float(np.prod(1 + window[-m:] / 100) - 1)
            else:
                row['sequence'] = window.astype('float32')
            if opts['sector']:
                sector = stoi.get(stock)
                if sector is None:
                    continue
                row['sector'] = sector
            if opts['covariates']:
                for name in COVARIATE_COLUMNS:
                    row[f'{name}_sequence'] = (
                        cov_z[name].to_numpy()[i - NUM_LAGS:i].astype('float32')
                    )
            rows.append(row)

    frame = pd.DataFrame(rows)
    ordered = ['stock', 'date', 'return']
    if opts['tabular']:
        ordered += ['return_t1'] + [f'cum_return_{m}' for m in TABULAR_HORIZONS[1:]]
    else:
        # Keep `sequence` first so the covariate models see the return series as
        # feature 0, matching the saved weights.
        ordered += ['sequence']
        ordered += [f'{c}_sequence' for c in COVARIATE_COLUMNS if opts['covariates']]
    ordered += ['Class0', 'Class1'] + (['sector'] if opts['sector'] else [])
    return frame[ordered].sort_values(['date', 'stock']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='lag_60',
                        help='which feature set to build (default: lag_60)')
    parser.add_argument('--years', nargs='*', type=int, default=list(STUDY_YEARS),
                        help='study periods to build (default: 1998-2024)')
    parser.add_argument('--standardisation', choices=['train', 'reference'],
                        default='train',
                        help="'train' recomputes mu/sigma from the training window; "
                             "'reference' reuses the constants of the published run")
    args = parser.parse_args()

    out_dir = SEQUENCES / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        frame = build_period(year, args.variant, args.standardisation)
        out = out_dir / f'returns_{year}.parquet'
        frame.to_parquet(out, index=False)
        print(f'{out}  {len(frame):,} windows, {frame.stock.nunique()} stocks')


if __name__ == '__main__':
    main()
