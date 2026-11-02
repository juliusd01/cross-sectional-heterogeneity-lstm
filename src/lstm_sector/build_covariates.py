"""Assemble data/covariates.csv from the raw vendor extracts.

The five macro-financial series of section 3.2 come from four providers and
cannot be redistributed, so neither the extracts nor the assembled file are part
of this repository.  This script documents exactly how they are combined, and
rebuilds the file for anyone holding the same subscriptions.

Expected layout under `--source-dir`:

    Bloomberg/oil_prices.csv                       date, PX_LAST   (WTI front month)
    Cboe/vix.csv                                   date, vix
    LSEG/gold_futures_GCc1.xls                     Date, SETTLE    (GCc1;
                                                   despite the extension this
                                                   export is CSV text)
    FederalReserveBoardReports/Interest_Rate_Daily.csv
                                                   date, t10y3m, dgs10

Two corrections from section 3.3 are applied here:

* WTI settled negative on 20 April 2020; that print is replaced by a linear
  interpolation of its neighbours.
* A handful of missing VIX observations between 1997 and 2021 are linearly
  interpolated; they show no pattern, so they are treated as missing at random.

Returns are left as fractions, not percent -- `build_sequences.py` standardises
every covariate per study period anyway, so only Figure 5 sees the raw scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import COVARIATES

OIL_ANOMALY = pd.Timestamp('2020-04-20')


def load_oil(source: Path) -> pd.Series:
    oil = pd.read_csv(source / 'Bloomberg' / 'oil_prices.csv',
                      parse_dates=['date'])
    oil = oil.drop_duplicates('date').set_index('date')['PX_LAST'].sort_index()
    if OIL_ANOMALY in oil.index:
        oil.loc[OIL_ANOMALY] = None
        oil = oil.interpolate(method='linear')
    return oil.pct_change(fill_method=None).rename('oil_return').dropna()


def load_gold(source: Path) -> pd.Series:
    path = source / 'LSEG' / 'gold_futures_GCc1.xls'
    # The LSEG export carries an .xls extension but is comma-separated text.
    try:
        gold = pd.read_csv(path, parse_dates=['Date'])
    except (UnicodeDecodeError, pd.errors.ParserError):
        gold = pd.read_excel(path, parse_dates=['Date'])
    gold = gold.drop_duplicates('Date').set_index('Date')['SETTLE'].sort_index()
    return gold.pct_change(fill_method=None).rename('gold_return').dropna()


def load_vix(source: Path) -> pd.Series:
    vix = pd.read_csv(source / 'Cboe' / 'vix.csv', parse_dates=['date'])
    vix = vix.drop_duplicates('date').set_index('date')['vix'].sort_index()
    return vix.interpolate(method='linear', limit_area='inside').rename('vix')


def load_rates(source: Path) -> tuple[pd.Series, pd.Series]:
    rates = pd.read_csv(
        source / 'FederalReserveBoardReports' / 'Interest_Rate_Daily.csv',
        parse_dates=['date'], usecols=['date', 't10y3m', 'dgs10'],
        low_memory=False)
    rates = rates.drop_duplicates('date').set_index('date').sort_index()
    t10y3m = rates['t10y3m'].dropna().rename('t10y3m')
    t10y_change = rates['dgs10'].dropna().diff().round(2).rename('t10y_change')
    return t10y3m, t10y_change.dropna()


def build(source: Path) -> pd.DataFrame:
    t10y3m, t10y_change = load_rates(source)
    series = [load_oil(source), load_gold(source), load_vix(source),
              t10y3m, t10y_change]
    cov = pd.concat(series, axis=1).sort_index()
    cov.index.name = 'date'
    return cov.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='directory holding the four vendor extracts')
    parser.add_argument('--out', type=Path, default=COVARIATES)
    args = parser.parse_args()

    cov = build(args.source_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(args.out, index=False)

    print(f'{len(cov):,} rows, {cov["date"].min().date()} to {cov["date"].max().date()}')
    for column in cov.columns[1:]:
        present = cov[column].notna()
        print(f'  {column:14s} {present.sum():6,} obs  '
              f'{cov.loc[present, "date"].min().date()} .. '
              f'{cov.loc[present, "date"].max().date()}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
