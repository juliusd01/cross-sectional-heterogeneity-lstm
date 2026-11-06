"""Tables 4 and 5: sector contributions against realised sector performance.

`--year 2007` reproduces Table 4, the case in which the learned industry
momentum pays off; `--year 2003` reproduces Table 5, the momentum-crash year in
which it inverts.  `--all-years` writes the per-year correlation series that
Figures 10 and 11 are built from.

Needs the raw price panels under data/prices/ -- see data/README.md.
"""
from __future__ import annotations

import argparse

import pandas as pd

from paths import TABLES, spec
from sector_analysis import (RETURN_MEASURES, compare, correlations,
                                         weighted_corr)

# The two case studies discussed in the paper, and their table numbers.
TABLE_NUMBER = {2007: 'tab04', 2003: 'tab05'}

PERCENT_COLUMNS = [
    'Mean_Return', 'Median_Return', 'Total_Return', 'Pct_Positive_Returns',
    'Train_Mean_Return_3Y', 'Train_Median_Return_3Y', 'Train_Total_Return_3Y',
]

DISPLAY_COLUMNS = [
    'Sector', 'Train_Median_Return_3Y', 'Difference_(z1-z0)', 'Mean_Return',
    'Median_Return', 'Pct_Positive_Returns', 'Total_Return', 'Num_Stocks',
]


def case_study(model_key: str, year: int) -> pd.DataFrame:
    merged = compare(spec(model_key).models, year)

    print(f'\n{"=" * 100}\nSECTOR ANALYSIS FOR {year}\n{"=" * 100}')
    print(merged[['Sector', 'Difference_(z1-z0)'] + RETURN_MEASURES
                 + ['Num_Stocks']].to_string(index=False))
    print('\nWeighted correlations with dC_c:')
    for measure, value in correlations(merged).items():
        print(f'  {measure:26s} {value:+.4f}')

    out = merged.copy()
    for column in PERCENT_COLUMNS:
        out[column] = (out[column] * 100).round(2)
    out['Difference_(z1-z0)'] = out['Difference_(z1-z0)'].round(2)
    return out


def all_years(model_key: str, start: int, end: int,
              annual_metrics_csv) -> pd.DataFrame:
    """Per-year dC_c correlation, joined to the sector-minus-base performance gap."""
    rows = []
    for year in range(start, end + 1):
        try:
            merged = compare(spec(model_key).models, year)
        except SystemExit as exc:
            print(f'warning: skipping {year}: {exc}')
            continue
        delta = merged['Difference_(z1-z0)'].values
        weights = merged['Num_Stocks'].values
        rows.append({
            'Year': year,
            'Corr_deltaC_s_Mean_Return': weighted_corr(
                delta, merged['Mean_Return'].values, weights),
            'Corr_deltaC_s_Median_Return': weighted_corr(
                delta, merged['Median_Return'].values, weights),
        })
    summary = pd.DataFrame(rows).sort_values('Year').reset_index(drop=True)

    metrics = pd.read_csv(annual_metrics_csv)
    metrics = metrics[metrics['model'].isin(['LSTM_Sector', 'LSTM_Base'])]
    wide = metrics.pivot_table(
        index='year', columns='model',
        values=['avg_accuracy', 'annualized_sharpe_ratio', 'avg_daily_return'],
        aggfunc='first')

    diffs = pd.DataFrame({
        'Year': wide.index,
        'Accuracy_Diff_Sector_minus_Base':
            (wide[('avg_accuracy', 'LSTM_Sector')]
             - wide[('avg_accuracy', 'LSTM_Base')]).values * 100,
        'Sharpe_Ratio_Diff_Sector_minus_Base':
            (wide[('annualized_sharpe_ratio', 'LSTM_Sector')]
             - wide[('annualized_sharpe_ratio', 'LSTM_Base')]).values,
        'Mean_Return_Diff_Sector_minus_Base':
            (wide[('avg_daily_return', 'LSTM_Sector')]
             - wide[('avg_daily_return', 'LSTM_Base')]).values,
    })
    return summary.merge(diffs, on='Year', how='left')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='sector_l3_d2',
                        help='sector-embedding spec key (default: sector_l3_d2)')
    parser.add_argument('--year', type=int, action='append',
                        help='study period for a case-study table; repeatable')
    parser.add_argument('--all-years', action='store_true',
                        help='write the per-year correlation series instead')
    parser.add_argument('--start', type=int, default=1998)
    parser.add_argument('--end', type=int, default=2024)
    parser.add_argument('--annual-metrics',
                        default=TABLES / 'annual_metrics_summary.csv')
    args = parser.parse_args()

    if args.all_years:
        summary = all_years(args.model, args.start, args.end, args.annual_metrics)
        out = TABLES / 'embedding_correlation_by_year.csv'
        summary.to_csv(out, index=False)
        summary.to_excel(TABLES / 'embedding_correlation_by_year.xlsx', index=False)
        print(summary.to_string(index=False))
        print(f'\nwrote {out}')
        return

    for year in args.year or [2007, 2003]:
        table = case_study(args.model, year)
        prefix = TABLE_NUMBER.get(year, 'sector_contribution')
        out = TABLES / f'{prefix}_sector_contribution_{year}.xlsx'
        table.to_excel(out, index=False)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
