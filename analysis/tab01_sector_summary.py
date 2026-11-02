"""Table 1: average yearly summary statistics for S&P 500 constituents by sector.

Equal-weighted portfolios per TRBC economic sector, formed from the index
constituents of each study period, 1995-2024.

Needs the raw price panels under data/prices/ -- see data/README.md.
"""
from __future__ import annotations

import argparse

from lstm_sector.paths import TABLES
from lstm_sector.sector_analysis import sector_summary_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--start', type=int, default=1995)
    parser.add_argument('--end', type=int, default=2024)
    args = parser.parse_args()

    table = sector_summary_statistics(args.start, args.end)
    out = TABLES / 'tab01_sector_summary_stats.xlsx'
    table.to_excel(out, index=False)
    print(table.to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
