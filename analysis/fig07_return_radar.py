"""Figure 7: return characteristics of the sector LSTM against three benchmarks.

Each spoke is min-max normalised across the four models, so a larger enclosed
area means better return characteristics.  All models except the buy-and-hold
market benchmark carry 2 bps of transaction costs per half turn.

`--scale volmatch` levers each strategy to the market's annualised volatility
first, so the return comparison is made at an equal risk budget -- matching the
convention used for Figure 8.

Reads the matching tables/tab07_return_characteristics*.csv,
writes figures/fig07_return_radar[_<scale>].png
"""
from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the 'science' style

from radar import draw, normalise, radar_factory
from paths import FIGURES, TABLES

plt.style.use('science')
plt.rcParams['text.usetex'] = False

MODELS = ['LSTM Base', 'LSTM Sector', 'Random Forest', 'Market']
COLORS = ['blue', 'red', 'green', 'orange']
METRICS = [
    ('mean_return', 'Mean Return'),
    ('sharpe_ratio', 'Sharpe Ratio'),
    ('annualized_return', 'Return p.a.'),
    ('excess_annualized_return', 'Excess return p.a.'),
    ('share_positive_returns', 'Share Positive Returns'),
    ('mean_return_long', 'Mean Return Long'),
    ('mean_return_short', 'Mean Return Short'),
]


# The performance table mixes units, so the axis labels cannot guess: the mean
# returns are already daily percentages, while the annualised figures and the
# positive-day share are stored as fractions.
FRACTION_METRICS = {'annualized_return', 'excess_annualized_return',
                    'share_positive_returns'}
DAILY_PERCENT_METRICS = {'mean_return', 'mean_return_long', 'mean_return_short'}


def format_range(metric: str, lo: float, hi: float) -> str:
    if metric == 'sharpe_ratio':
        return f'{lo:.2f} - {hi:.2f}'
    if metric in FRACTION_METRICS:
        return f'{lo * 100:.1f}% - {hi * 100:.1f}%'
    if metric in DAILY_PERCENT_METRICS:
        # already daily percentages, so printed as-is rather than rescaled
        return f'{lo:.2f}% - {hi:.2f}%'
    return f'{lo:.1f} - {hi:.1f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scale', default='volmatch',
                        choices=('percent', 'volmatch', 'sigma'),
                        help='scaling of the strategy returns (default: percent)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    suffix = '' if args.scale == 'volmatch' else f'_{args.scale}'
    if args.out is None:
        args.out = FIGURES / f'fig07_return_radar{suffix}.png'
    table = TABLES / f'tab07_return_characteristics{suffix}.csv'
    if not table.exists():
        raise SystemExit(f'{table} not found -- run:\n'
                         f'  python analysis/tab07_performance.py --scale {args.scale}')
    performance = pd.read_csv(table, index_col=0)

    per_metric = {m: [performance.loc[m, model] for model in MODELS]
                  for m, _ in METRICS}
    radar_data = [[normalise(per_metric[m])[i] for m, _ in METRICS]
                  for i in range(len(MODELS))]
    labels = [f'{label}\n({format_range(m, min(per_metric[m]), max(per_metric[m]))})'
              for m, label in METRICS]

    theta = radar_factory(len(METRICS), frame='polygon')
    note = ('strategies levered to market volatility'
            if args.scale == 'volmatch' else None)
    fig = draw(theta, radar_data, MODELS, labels, COLORS, note=note)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
