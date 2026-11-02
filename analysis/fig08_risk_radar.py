"""Figure 8: risk characteristics of the sector LSTM against three benchmarks.

Spokes carry the absolute value of each risk measure and are normalised so that
a larger enclosed area means more risk -- the opposite reading of Figure 7.

Drawn at k = 10 with every strategy levered to the market's annualised
volatility, so the comparison is made at one risk budget rather than being
driven by the long-short portfolio holding 20 names against the index's ~500.
Total volatility is then equal by construction, so that spoke carries downside
deviation instead.

Reads the matching tables/tab07_return_characteristics*.csv,
writes figures/fig08_risk_radar.png
"""
from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the 'science' style

from lstm_sector.radar import draw, normalise, radar_factory
from lstm_sector.paths import FIGURES, TABLES

plt.style.use('science')
plt.rcParams['text.usetex'] = False

MODELS = ['LSTM Base', 'LSTM Sector', 'Random Forest', 'Market']
COLORS = ['blue', 'red', 'green', 'orange']
METRICS = [
    ('one_percent_VaR', '1% VaR'),
    ('five_percent_VaR', '5% VaR'),
    ('one_percent_CVaR', '1% CVaR'),
    ('five_percent_CVaR', '5% CVaR'),
    ('max_drawdown', 'Max Drawdown'),
    ('annualized_volatility', 'Ann. Volatility'),   # swapped out when vol-matched
    ('min_return', 'Max Daily Loss'),
    ('kurtosis', 'Kurtosis'),
]

NOTE = ('Smaller area = lower risk (better)  •  '
        'Larger area = higher risk (worse)')


def format_range(metric: str, lo: float, hi: float) -> str:
    if metric == 'max_drawdown':
        return f'{lo * 100:.0f}% — {hi * 100:.0f}%'
    if metric == 'kurtosis':
        return f'{lo:.1f} — {hi:.1f}'
    if metric in ('annualized_volatility', 'downside_deviation', 'min_return'):
        return f'{lo:.1f}% — {hi:.1f}%'
    return f'{lo:.2f}% — {hi:.2f}%'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--k', type=int, default=10,
                        help='positions per leg behind the table to plot')
    parser.add_argument('--scale', default='volmatch',
                        choices=('percent', 'volmatch', 'sigma'),
                        help='scaling of the strategy returns (default: percent)')
    parser.add_argument('--out', default=None,
                        help='default: figures/fig08_risk_radar_k<k>[_<scale>].png')
    args = parser.parse_args()

    suffix = '' if args.k == 10 else f'_k{args.k}'
    if args.scale != 'volmatch':
        suffix += f'_{args.scale}'
    if args.out is None:
        args.out = FIGURES / f'fig08_risk_radar_k{args.k}{"" if args.scale == "volmatch" else "_" + args.scale}.png'
    table = TABLES / f'tab07_return_characteristics{suffix}.csv'
    if not table.exists():
        raise SystemExit(f'{table} not found -- run:\n'
                         f'  python analysis/tab07_performance.py '
                         f'--k {args.k} --scale {args.scale}')
    performance = pd.read_csv(table, index_col=0)

    metrics = list(METRICS)
    if args.scale == 'volmatch':
        # Total volatility is equal by construction once the strategies are
        # levered to the market's; downside deviation carries the same units
        # and still discriminates.
        metrics = [('downside_deviation', 'Downside Deviation')
                   if m == 'annualized_volatility' else (m, label)
                   for m, label in metrics]

    per_metric = {m: [abs(performance.loc[m, model]) for model in MODELS]
                  for m, _ in metrics}
    radar_data = [[normalise(per_metric[m])[i] for m, _ in metrics]
                  for i in range(len(MODELS))]
    labels = [f'{label}\n({format_range(m, min(per_metric[m]), max(per_metric[m]))})'
              for m, label in metrics]

    theta = radar_factory(len(metrics), frame='polygon')
    scale_note = {'percent': '', 'volmatch': '  •  strategies levered to market volatility',
                  'sigma': '  •  strategies in training-sigma units'}[args.scale]
    note = f'{NOTE}  •  k = {args.k} ({2 * args.k} positions){scale_note}'
    fig = draw(theta, radar_data, MODELS, labels, COLORS, note=note)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
