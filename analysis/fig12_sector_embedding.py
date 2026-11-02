"""Figure 12: the learned sector embedding vectors in two dimensions.

Plots the rows of W_emb for one study period as vectors from the origin.
Sectors that the network treats alike during training end up pointing in
similar directions.

Reads the fitted weights in models/; no raw price data required.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the 'science' style

from lstm_sector.paths import FIGURES, SECTORS, SECTORS_11, spec
from lstm_sector.sector_analysis import load_state_dict

plt.style.use('science')
plt.rcParams['text.usetex'] = False


def plot(weights: np.ndarray, names: list[str], year: int, out_png) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for (x, y), color, sector in zip(weights, colors, names):
        ax.quiver(0, 0, x, y, color=color, angles='xy', scale_units='xy',
                  scale=1, width=0.005, alpha=0.8)
        ax.scatter(x, y, color=color, s=100, alpha=0.8,
                   edgecolors='black', linewidth=1)
        ax.text(x, y, f'  {sector}', fontsize=9, ha='left')

    ax.scatter(0, 0, color='black', s=100, marker='o', label='Origin')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title(f'2D Visualization of Sector Embedding Vectors ({year})',
                 fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    limit = np.abs(weights).max() * 1.15
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_aspect('equal')

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='sector_l3_d2')
    parser.add_argument('--year', type=int, default=2007)
    args = parser.parse_args()

    state = load_state_dict(spec(args.model).models, args.year)
    weights = state['sector_embedding.weight'].cpu().numpy()
    if weights.shape[1] != 2:
        raise SystemExit(f'{args.model} has a {weights.shape[1]}-dimensional '
                         'embedding; this figure needs a 2-dimensional one')

    names = SECTORS_11 if weights.shape[0] == 11 else SECTORS
    prefix = 'fig12' if args.year == 2007 else 'sector_embedding'
    out = FIGURES / f'{prefix}_sector_embedding_{args.year}.png'
    plot(weights, names, args.year, out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
