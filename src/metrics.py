"""Daily long-short portfolio metrics.

`build()` turns the per-stock prediction files written by the training scripts
into one row per trading day: the k stocks held long, the k held short, their
realised returns, the transaction cost incurred, and the resulting portfolio
return.  Every table and figure in the paper is computed from these files.

`load()` is what the analysis scripts call.  It reads the committed parquet
files under metrics/ and falls back to building them from predictions/ when a
combination of (spec, k, transaction cost) has not been materialised.
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA, PREDICTIONS, Spec, spec as lookup_spec

LIST_COLUMNS = ['flop_stocks', 'top_stocks']

# The archived prediction files store the realised return in standardised units
# rather than percent, which understated every reported return by a factor of
# sigma.  This marker tells `build()` to undo that standardisation; prediction
# files produced by the current build_sequences.py are already in percent and
# carry no marker.  See README, "Return units".
LEGACY_MARKER = PREDICTIONS / 'RETURNS_ARE_STANDARDISED'


def _standardisation_constants() -> pd.DataFrame:
    return pd.read_csv(DATA / 'standardisation_reference.csv').set_index('study_period')


def to_percent(df: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Undo the training-window standardisation: r = z * sigma + mu."""
    row = _standardisation_constants().loc[int(year)]
    out = df.copy()
    out['return'] = out['return'] * row['sigma_train_pct'] + row['mu_train_pct']
    return out


def _filename(k: int, trading_cost_pct: float) -> str:
    """metrics_k10.csv / metrics_k10_tc2bps.csv, the naming used by the paper."""
    suffix = f'_tc{int(trading_cost_pct * 100)}bps' if trading_cost_pct > 0 else ''
    return f'metrics_k{k}{suffix}'


def select_top_flop(k: int, data: pd.DataFrame, unique_dates) -> dict:
    """Pick, for each date, the k stocks with the highest class-0 and class-1
    probability.  Ties are broken by a date-seeded random draw so that the
    selection is deterministic but not biased by the row order of the input."""
    selection = {}
    for i, date in enumerate(unique_dates):
        date_data = data[data['date'] == date]

        top_class0 = date_data.nlargest(k, 'pred_Class0', keep='all')
        if len(top_class0) > k:
            top_class0 = top_class0.sample(n=k, random_state=43 + i)
        top_class0 = top_class0[['stock', 'pred_Class0', 'pred_Class1', 'return',
                                 'predicted_class', 'actual_class']]

        top_class1 = date_data.nlargest(k, 'pred_Class1', keep='all')
        if len(top_class1) > k:
            top_class1 = top_class1.sample(n=k, random_state=42 + i)
        top_class1 = top_class1[['stock', 'pred_Class1', 'pred_Class0', 'return',
                                 'predicted_class', 'actual_class']]

        selection[date] = {'top_class0': top_class0, 'top_class1': top_class1}
    return selection


def _year_metrics(results_df: pd.DataFrame, year: str, k: int,
                  trading_cost_pct: float) -> pd.DataFrame:
    unique_dates = sorted(results_df['date'].unique())
    selection = select_top_flop(k, results_df, unique_dates)

    rows = []
    for i, date in enumerate(unique_dates):
        flop_k = selection[date]['top_class0']
        avg_return_flop_k = flop_k['return'].mean()
        accuracy_flop_k = 1 - flop_k['actual_class'].sum() / k
        flop_stocks = flop_k['stock'].tolist()

        top_k = selection[date]['top_class1']
        avg_return_top_k = top_k['return'].mean()
        accuracy_top_k = top_k['actual_class'].sum() / k
        top_stocks = top_k['stock'].tolist()

        # A position is only charged when it is opened and when it is closed.
        # A stock held on consecutive days therefore costs nothing in between.
        current_top, current_flop = set(top_stocks), set(flop_stocks)
        if i > 0:
            prev = selection[unique_dates[i - 1]]
            prev_top = set(prev['top_class1']['stock'].tolist())
            prev_flop = set(prev['top_class0']['stock'].tolist())
        else:
            prev_top, prev_flop = set(), set()
        if i < len(unique_dates) - 1:
            nxt = selection[unique_dates[i + 1]]
            next_top = set(nxt['top_class1']['stock'].tolist())
            next_flop = set(nxt['top_class0']['stock'].tolist())
        else:
            next_top, next_flop = set(), set()

        num_long_trades = len(current_top - prev_top) + len(current_top - next_top)
        num_short_trades = len(current_flop - prev_flop) + len(current_flop - next_flop)

        # Each traded name carries 1/k of the leg, so the per-stock cost is
        # divided by k.  (The original code hardcoded 10, which is correct only
        # at k = 10 -- the size used everywhere in the paper -- and overstated
        # costs tenfold at k = 100.)
        long_trading_cost = num_long_trades * trading_cost_pct / k
        short_trading_cost = num_short_trades * trading_cost_pct / k

        avg_return_top_k_net = avg_return_top_k - long_trading_cost
        # added rather than subtracted because the leg is short
        avg_return_flop_k_net = avg_return_flop_k + short_trading_cost

        rows.append({
            'year': year,
            'date': date,
            'avg_return_flop_k': -avg_return_flop_k_net,
            'accuracy_flop_k': accuracy_flop_k,
            'avg_return_top_k': avg_return_top_k_net,
            'accuracy_top_k': accuracy_top_k,
            'trading_cost': long_trading_cost + short_trading_cost,
            'avg_daily_return': avg_return_top_k_net - avg_return_flop_k_net,
            'accuracy': round((accuracy_flop_k + accuracy_top_k) / 2, 2),
            'flop_stocks': flop_stocks,
            'top_stocks': top_stocks,
        })
    return pd.DataFrame(rows)


def build(spec: Spec | str, k: int, trading_cost_pct: float = 0.0,
          write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Compute daily metrics for one specification from its prediction files."""
    spec = lookup_spec(spec) if isinstance(spec, str) else spec
    files = sorted(glob.glob(str(spec.predictions / 'predictions_*.parquet')))
    if not files:
        raise SystemExit(
            f'no prediction files in {spec.predictions}.\n'
            'The full prediction set is not committed -- see README, "Artifacts".'
        )

    legacy = LEGACY_MARKER.exists()
    frames = []
    for path in files:
        year = Path(path).stem.split('_')[-1]
        if verbose:
            print(f'  {spec.key} k={k} tc={trading_cost_pct}: {year}')
        predictions = pd.read_parquet(path)
        if legacy:
            predictions = to_percent(predictions, year)
        frames.append(_year_metrics(predictions, year, k, trading_cost_pct))

    metrics_df = pd.concat(frames, ignore_index=True)
    if write:
        spec.metrics.mkdir(parents=True, exist_ok=True)
        out = spec.metrics / f'{_filename(k, trading_cost_pct)}.parquet'
        _write_parquet(metrics_df, out)
        if verbose:
            print(f'wrote {out}')
    return metrics_df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Store metrics compactly: zstd plus narrow dtypes keeps a full 27-year
    k=10 series around 350 kB instead of the 2.2 MB the CSV takes."""
    out = df.copy()
    out['year'] = out['year'].astype('int16')
    out['date'] = pd.to_datetime(out['date'])
    for col in out.select_dtypes('float64'):
        out[col] = out[col].astype('float32')
    out.to_parquet(path, index=False, compression='zstd', compression_level=19)


def load(spec: Spec | str, k: int = 10, trading_cost_pct: float = 0.0,
         lists: bool = False) -> pd.DataFrame:
    """Return the daily metrics for one specification.

    Reads the committed parquet file; builds it from predictions/ if missing.
    Pass `lists=True` when the per-day ticker lists are needed -- they are
    dropped otherwise because they dominate memory for large k.
    """
    spec = lookup_spec(spec) if isinstance(spec, str) else spec
    name = _filename(k, trading_cost_pct)

    parquet = spec.metrics / f'{name}.parquet'
    csv = spec.predictions / f'{name}.csv'
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        df = build(spec, k, trading_cost_pct, write=True, verbose=False)

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['year'].astype(int)
    if lists:
        for col in LIST_COLUMNS:
            if col in df:
                df[col] = df[col].map(
                    lambda v: ast.literal_eval(v) if isinstance(v, str) else list(v)
                )
    else:
        df = df.drop(columns=[c for c in LIST_COLUMNS if c in df])
    return df


def cross_sectional_accuracy(spec: Spec | str) -> pd.DataFrame:
    """Classification accuracy over the whole cross-section, by year.

    This is the `Acc_all` column of Table 6.  It needs the raw prediction
    files, so the values are cached in metrics/accuracy_all.csv for clones that
    do not download them.
    """
    spec = lookup_spec(spec) if isinstance(spec, str) else spec
    rows = []
    for path in sorted(glob.glob(str(spec.predictions / 'predictions_*.parquet'))):
        df = pd.read_parquet(path, columns=['date', 'predicted_class', 'actual_class'])
        rows.append({
            'spec': spec.key,
            'year': int(Path(path).stem.split('_')[-1]),
            'n_obs': len(df),
            'accuracy_all': float((df['predicted_class'] == df['actual_class']).mean()),
        })
    return pd.DataFrame(rows)


def risk_free_rate() -> pd.Series:
    """Daily risk-free rate from the Kenneth R. French data library, in percent."""
    from paths import MARKET_RETURN
    market = pd.read_csv(MARKET_RETURN)
    market['date'] = pd.to_datetime(market['date'], format='%Y%m%d')
    return market.set_index('date')['rf']


def market_return() -> pd.DataFrame:
    """Daily value-weighted US market return (mktrf + rf), in percent."""
    from paths import MARKET_RETURN
    market = pd.read_csv(MARKET_RETURN)
    market['date'] = pd.to_datetime(market['date'], format='%Y%m%d')
    market['return'] = market['mkt-rf'] + market['rf']
    return market


def annualised_sharpe(returns: pd.Series, rf: pd.Series | float = 0.0) -> float:
    """Sharpe ratio of a daily return series, annualised by sqrt(252)."""
    from paths import TRADING_DAYS
    excess_ann = (np.mean(returns) - np.mean(rf)) * TRADING_DAYS
    vol_ann = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)
    return excess_ann / vol_ann
