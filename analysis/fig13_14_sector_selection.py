"""Figures 13 and 14: what the sector embeddings do to stock selection.

For one test year, compares the sector composition of the long leg with that of
the short leg of the k=10 portfolio, against the baseline composition of the
index universe.  A large positive bar means the sector is heavily over-weighted
in the long leg -- Energy in 2007 (Figure 13), Utilities in 2003 (Figure 14).

Needs the raw price panels under data/prices/ -- see data/README.md.
"""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from lstm_sector.metrics import load
from lstm_sector.paths import COMPANY_SECTOR, FIGURES, PRICES, TABLES, spec

TABLE_NUMBER = {2007: 'fig13', 2003: 'fig14'}


def ticker_to_sector() -> dict:
    sectors = pd.read_csv(COMPANY_SECTOR, usecols=['Instrument', 'Economic_Sector'])
    return dict(zip(sectors['Instrument'], sectors['Economic_Sector']))


def baseline_composition(year: int, stoi: dict) -> tuple[Counter, dict]:
    """Sector composition of the index universe for this study period."""
    path = PRICES / f'test_{year}.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found.  The Refinitiv price panels are not redistributable;\n'
            'see data/README.md for the required format.'
        )
    tickers = [c for c in pd.read_csv(path, nrows=0).columns if c != 'Date']
    counts = Counter(stoi[t] for t in tickers if t in stoi)
    total = sum(counts.values())
    return counts, {s: c / total * 100 for s, c in counts.items()}


def leg_composition(metrics_year: pd.DataFrame, column: str,
                    stoi: dict) -> Counter:
    return Counter(stoi[t] for row in metrics_year[column]
                   for t in row if t in stoi)


def comparison(year: int, model_key: str) -> pd.DataFrame:
    stoi = ticker_to_sector()
    baseline_counts, baseline_pct = baseline_composition(year, stoi)

    metrics = load(spec(model_key), k=10, lists=True)
    metrics_year = metrics[metrics['year'] == year]
    if metrics_year.empty:
        raise SystemExit(f'no metrics rows for year {year}')

    flop = leg_composition(metrics_year, 'flop_stocks', stoi)
    top = leg_composition(metrics_year, 'top_stocks', stoi)
    n_flop, n_top = sum(flop.values()), sum(top.values())

    rows = []
    for sector in set(baseline_counts) | set(flop) | set(top):
        flop_pct = flop.get(sector, 0) / n_flop * 100 if n_flop else 0
        top_pct = top.get(sector, 0) / n_top * 100 if n_top else 0
        rows.append({
            'Sector': sector,
            'Baseline_Count': baseline_counts.get(sector, 0),
            'Baseline_%': baseline_pct.get(sector, 0),
            'Flop_Count': flop.get(sector, 0),
            'Flop_%': flop_pct,
            'Top_Count': top.get(sector, 0),
            'Top_%': top_pct,
            'Flop_vs_Baseline_%': flop_pct - baseline_pct.get(sector, 0),
            'Top_vs_Baseline_%': top_pct - baseline_pct.get(sector, 0),
            # positive = over-represented in the long leg
            'Top_vs_Flop_%': top_pct - flop_pct,
        })
    return pd.DataFrame(rows).sort_values('Top_vs_Flop_%', ascending=False)


def plot(table: pd.DataFrame, year: int, out_png) -> None:
    ordered = table.sort_values('Top_vs_Flop_%', ascending=True)
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in ordered['Top_vs_Flop_%']]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.barh(range(len(ordered)), ordered['Top_vs_Flop_%'], color=colors, alpha=0.9)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered['Sector'], fontsize=10)
    ax.set_xlabel('Percentage-point difference (Top % - Flop %)')
    ax.set_title(f'Top vs Flop Sector Bias, {year} '
                 '(green = more top, red = more flop)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.9)
    ax.grid(axis='x', alpha=0.3)

    # headroom so the value labels stay inside the axes
    values = ordered['Top_vs_Flop_%']
    ax.set_xlim(values.min() - 8.0, values.max() + 8.0)
    for i, value in enumerate(values):
        ax.text(value + (0.7 if value >= 0 else -0.7), i, f'{value:+.1f}%',
                ha='left' if value >= 0 else 'right', va='center', fontsize=9)

    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.22, right=0.99)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='sector_l3_d2')
    parser.add_argument('--year', type=int, action='append',
                        help='test year; repeatable (default: 2007 and 2003)')
    args = parser.parse_args()

    for year in args.year or [2007, 2003]:
        table = comparison(year, args.model)
        prefix = TABLE_NUMBER.get(year, 'sector_selection')
        csv = TABLES / f'sector_selection_{year}.csv'
        png = FIGURES / f'{prefix}_sector_selection_{year}.png'
        table.to_csv(csv, index=False)
        plot(table, year, png)
        print(f'\n=== sector selection bias, {year} ===')
        print(table.to_string(index=False))
        print(f'wrote {csv}\nwrote {png}')


if __name__ == '__main__':
    main()
