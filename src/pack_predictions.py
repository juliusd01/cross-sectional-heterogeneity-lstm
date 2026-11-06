"""Pack predictions/ into a compact archive for a GitHub Release or Zenodo deposit.

The raw prediction files are ~1.5 GB and are not committed.  Everything the
paper reports is derived from metrics/, which is committed, so the archive is
only needed by someone who wants to recompute metrics at a portfolio size or
transaction cost that is not already materialised.

Repacking with narrow dtypes and zstd cuts the set to roughly half its size
without losing anything the metrics code reads.  `--drop-returns` removes the
realised `return` column as well, for the case where redistributing realised
returns of index constituents would breach a data licence; note that the
resulting files can no longer be used to recompute portfolio performance, only
prediction accuracy.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from paths import PREDICTIONS, SPECS

KEEP = ['stock', 'date', 'return', 'pred_Class0', 'pred_Class1',
        'predicted_class', 'actual_class']


def slim(df: pd.DataFrame, drop_returns: bool) -> pd.DataFrame:
    out = df[[c for c in KEEP if c in df.columns]].copy()
    if drop_returns:
        out = out.drop(columns=['return'], errors='ignore')
    elif 'return' in out:
        out['return'] = out['return'].astype('float32')
    out['stock'] = out['stock'].astype('category')
    for column in ('predicted_class', 'actual_class'):
        if column in out:
            out[column] = out[column].astype('int8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=Path('predictions_packed'),
                        help='directory to write the repacked tree into')
    parser.add_argument('--specs', nargs='*',
                        help='spec keys to include (default: all)')
    parser.add_argument('--drop-returns', action='store_true',
                        help='omit the realised return column')
    args = parser.parse_args()

    specs = [s for s in SPECS if not args.specs or s.key in args.specs]
    total_in = total_out = 0

    for spec in specs:
        files = sorted(glob.glob(str(spec.predictions / 'predictions_*.parquet')))
        if not files:
            print(f'skipping {spec.key}: nothing in {spec.predictions}')
            continue
        target = args.out / spec.path
        target.mkdir(parents=True, exist_ok=True)
        for path in files:
            path = Path(path)
            out_path = target / path.name
            slim(pd.read_parquet(path), args.drop_returns).to_parquet(
                out_path, index=False, compression='zstd', compression_level=19)
            total_in += path.stat().st_size
            total_out += out_path.stat().st_size
        print(f'{spec.key:14s} {len(files)} files -> {target}')

    if total_in:
        print(f'\n{total_in / 1e9:.2f} GB -> {total_out / 1e9:.2f} GB '
              f'({total_out / total_in:.0%})')
        print(f'archive with:  tar -caf predictions.tar.zst -C {args.out.parent} '
              f'{args.out.name}')


if __name__ == '__main__':
    main()
