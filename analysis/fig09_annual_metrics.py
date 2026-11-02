"""Figure 9: long-term validation -- performance metrics by study period.

Mean daily return, annualised Sharpe ratio and directional accuracy of the
k=10 long-short portfolio, before transaction costs, for each of the 27 test
years.  This is the evidence for the deterioration after 2008.

Writes tables/annual_metrics_summary.csv and figures/fig09_annual_metrics.png.
The CSV is also the input the embedding-quality analysis (Figures 10 and 11)
uses to relate the sector LSTM's edge over the base LSTM to the embeddings.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lstm_sector.metrics import load
from lstm_sector.paths import FIGURES, TABLES, TRADING_DAYS, spec

# Names are used verbatim in the CSV and read back by fig10_11.
MODELS = {
    'LSTM_Base': 'base_l3',
    'LSTM_Sector': 'sector_l3_d2',
    'Random_Forest': 'rf_n500',
}

PANELS = [
    ('avg_daily_return', 'Average Daily Return (%)',
     'Average Daily Returns by Year', 'o', None),
    ('annualized_sharpe_ratio', 'Annualized Sharpe Ratio',
     'Annualized Sharpe Ratio by Year', 's', None),
    ('avg_accuracy', 'Average Accuracy',
     'Average Accuracy by Year', '^', (0.485, 0.575)),
]


def annual_metrics(trading_cost_pct: float = 0.0) -> pd.DataFrame:
    """One row per model and study period."""
    frames = []
    for label, key in MODELS.items():
        df = load(spec(key), k=10, trading_cost_pct=trading_cost_pct)
        stats = df.groupby('year').agg(
            avg_daily_return=('avg_daily_return', 'mean'),
            std_daily_return=('avg_daily_return', 'std'),
            avg_accuracy=('accuracy', 'mean'),
        ).reset_index()
        stats['annualized_sharpe_ratio'] = (
            stats['avg_daily_return'] / stats['std_daily_return']
        ) * np.sqrt(TRADING_DAYS)
        stats['annualized_sharpe_ratio'] = stats['annualized_sharpe_ratio'].replace(
            [np.inf, -np.inf], np.nan)
        stats['model'] = label
        frames.append(stats[['model', 'year', 'avg_daily_return',
                             'annualized_sharpe_ratio', 'avg_accuracy']])

    return (pd.concat(frames, ignore_index=True)
            .sort_values(['model', 'year']).reset_index(drop=True))


def plot(df: pd.DataFrame, out_png) -> None:
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    models = df['model'].unique()
    colors = sns.color_palette('husl', len(models))

    for ax, (column, ylabel, title, marker, ylim) in zip(axes, PANELS):
        for color, model in zip(colors, models):
            series = df[df['model'] == model]
            ax.plot(series['year'], series[column], marker=marker,
                    label=model, color=color, linewidth=2)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        if ylim is None:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        else:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trading-cost', type=float, default=0.0,
                        help='cost per half turn in percent (default: 0, gross)')
    parser.add_argument('--out-csv', default=TABLES / 'annual_metrics_summary.csv')
    parser.add_argument('--out-png', default=FIGURES / 'fig09_annual_metrics.png')
    args = parser.parse_args()

    df = annual_metrics(args.trading_cost)
    df.to_csv(args.out_csv, index=False)
    plot(df, args.out_png)
    print(f'wrote {args.out_csv}\nwrote {args.out_png}')


if __name__ == '__main__':
    main()
