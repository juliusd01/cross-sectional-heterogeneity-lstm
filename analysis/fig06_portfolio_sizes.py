"""Figure 6: daily performance of the long-short portfolio for different sizes k.

Mean return, standard deviation, annualised Sharpe ratio and accuracy for the
four headline specifications, pooled over 1998-2024, net of 2 bps transaction
costs per half turn.  The Sharpe ratio is computed on returns in excess of the
risk-free rate, so it agrees with Tables 3 and 7 at k = 10.

Writes tables/portfolio_size_metrics_daily.csv and figures/fig06_daily_performance_by_k.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lstm_sector.metrics import load, risk_free_rate
from lstm_sector.paths import FIGURES, MAIN_LABELS, TABLES, TRADING_DAYS, spec

K_LIST = [5, 10, 50, 100, 200]
PLOT_K = [5, 10, 50, 100]          # k=200 is tabulated but crowds the figure
METRICS = ['Return (%)', 'Standard deviation (%)', 'Sharpe Ratio', 'Accuracy']
COLORS = ['#0086d6', '#8ecae6', '#005f73', '#48cae4']
MODEL_ORDER = ['Sector LSTM', 'Base LSTM', 'Cov LSTM', 'RF']


def build_table(trading_cost_pct: float = 0.02) -> pd.DataFrame:
    rf = risk_free_rate()
    rows = []
    for key, label in MAIN_LABELS.items():
        for k in K_LIST:
            df = load(spec(key), k=k, trading_cost_pct=trading_cost_pct)
            df = df.merge(rf, left_on='date', right_index=True, how='left')
            # Sharpe on excess returns, matching the Ledoit-Wolf test and Table 7
            excess = df['avg_daily_return'] - df['rf']
            rows.append({
                'model': label,
                'Return (%)': df['avg_daily_return'].mean(),
                'Standard deviation (%)': df['avg_daily_return'].std(),
                'Sharpe Ratio': (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS),
                'Accuracy': df['accuracy'].mean(),
                'k': k,
            })
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, out_png) -> None:
    long = table.melt(id_vars=['model', 'k'], value_vars=METRICS)
    long = long[long['k'].isin(PLOT_K)].copy()
    long['model'] = long['model'].str.replace(' ', '\n')
    order = [m.replace(' ', '\n') for m in MODEL_ORDER]

    sns.set_style('ticks')
    g = sns.FacetGrid(long, row='variable', col='k', sharey='row', sharex=True,
                      margin_titles=True, height=2, aspect=1.5)
    g.map_dataframe(sns.barplot, x='model', y='value', hue='model', dodge=False,
                    palette=COLORS, order=order, width=0.5)

    for ax, row_name in zip(g.axes[:, 0], METRICS):
        ax.set_ylabel(row_name, fontweight='bold', fontsize=10)
    g.set_titles(col_template='k={col_name}', row_template='')
    g.set_axis_labels(x_var='')
    g.despine(left=True)
    g.set(yticks=[])

    # Zoom each row onto its own data range; the differences between models are
    # small relative to the level.
    for i, metric in enumerate(METRICS):
        values = long.loc[long['variable'] == metric, 'value']
        lo, hi = values.min(), values.max()
        span = hi - lo
        for ax in g.axes[i, :]:
            ax.set_ylim(0 if metric == 'Sharpe Ratio' else lo - 0.1 * span,
                        hi + 0.15 * span)

    for ax in g.axes.flat:
        for patch in ax.patches:
            if patch.get_height() > 0:
                ax.annotate(f'{patch.get_height():.2f}',
                            (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                            ha='center', va='bottom', fontsize=8,
                            xytext=(0, 2), textcoords='offset points')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trading-cost', type=float, default=0.02,
                        help='cost per half turn in percent (default: 0.02 = 2 bps)')
    parser.add_argument('--out-csv', default=TABLES / 'portfolio_size_metrics_daily.csv')
    parser.add_argument('--out-png', default=FIGURES / 'fig06_daily_performance_by_k.png')
    args = parser.parse_args()

    table = build_table(args.trading_cost)
    table.to_csv(args.out_csv, index=False)
    plot(table, args.out_png)
    print(table.to_string(index=False))
    print(f'\nwrote {args.out_csv}\nwrote {args.out_png}')


if __name__ == '__main__':
    main()
