"""Table 2: pairwise one-sided Diebold-Mariano tests of predictive accuracy.

The loss series stacks the daily misclassification rate of the long leg and of
the short leg, giving two observations per trading day.  A rejection in favour
of the row model means it forecasts significantly better than the column model.

Writes tables/tab02_diebold_mariano.csv
"""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from metrics import load
from paths import TABLES, spec

# Accuracy does not depend on transaction costs, so the choice of cost series
# here is immaterial; 2 bps is used to match the rest of the paper.
DEFAULT_MODELS = {
    "Sector LSTM": "sector_l3_d2",
    "Cov LSTM": "cov_l1",
    "Base LSTM": "base_l3",
    "RF": "rf_n500",
}


def load_metric_file(key: str) -> pd.DataFrame:
    df = load(spec(key), k=10, trading_cost_pct=0.02)
    return df[["date", "accuracy_flop_k", "accuracy_top_k"]]


def stacked_loss_series(df: pd.DataFrame) -> np.ndarray:
    flop_loss = 1.0 - df["accuracy_flop_k"].astype(float).to_numpy()
    top_loss = 1.0 - df["accuracy_top_k"].astype(float).to_numpy()
    loss = np.empty(flop_loss.size * 2, dtype=float)
    loss[0::2] = flop_loss
    loss[1::2] = top_loss
    return loss


def newey_west_lrv(diff: np.ndarray, lag: int | None) -> tuple[float, int]:
    x = np.asarray(diff, dtype=float)
    n = x.size
    centered = x - x.mean()

    if lag is None:
        lag = int(np.floor(n ** (1 / 3)))
    lag = max(0, min(lag, n - 1))

    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1)
        gamma_l = float(np.dot(centered[l:], centered[:-l]) / n)
        lrv += 2.0 * weight * gamma_l
    return lrv, lag


def dm_table(models: dict[str, str], lag: int | None) -> pd.DataFrame:
    raw = {name: load_metric_file(key) for name, key in models.items()}

    common_dates = None
    for df in raw.values():
        dates = set(df["date"])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates)

    aligned_losses: dict[str, np.ndarray] = {}
    for name, df in raw.items():
        aligned = df[df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
        aligned_losses[name] = stacked_loss_series(aligned)

    obs = len(next(iter(aligned_losses.values())))
    rows: list[dict[str, object]] = []

    for i, j in permutations(aligned_losses.keys(), 2):
        diff = aligned_losses[i] - aligned_losses[j]
        mean_diff = float(diff.mean())
        lrv, used_lag = newey_west_lrv(diff, lag)
        se = np.sqrt(lrv / obs) if lrv > 0 else np.nan
        dm_stat = mean_diff / se if np.isfinite(se) and se > 0 else np.nan
        # H0: model_i inferior or equal accuracy to model_j (E[d] >= 0), H1: E[d] < 0
        pval = float(NormalDist().cdf(dm_stat)) if np.isfinite(dm_stat) else np.nan

        rows.append(
            {
                "model_i": i,
                "model_j": j,
                "n_days_common": len(common_dates),
                "n_obs_day_side": obs,
                "mean_loss_diff_i_minus_j": mean_diff,
                "dm_stat": dm_stat,
                "pvalue_H0_i_inferior": pval,
                "reject_H0_5pct": bool(pval < 0.05) if np.isfinite(pval) else False,
                "nw_lag": used_lag,
            }
        )

    return pd.DataFrame(rows).sort_values(["model_i", "model_j"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        action="append",
        required=False,
        help="Model as label=spec_key (see paths.SPECS). Repeat per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES / "tab02_diebold_mariano.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=None,
        help="Newey-West lag. Default: floor(n^(1/3)).",
    )
    args = parser.parse_args()

    if args.model:
        models = dict(pair.split("=", 1) for pair in args.model)
        if len(models) < 2:
            raise SystemExit("provide at least two --model entries")
    else:
        models = DEFAULT_MODELS

    table = dm_table(models, args.lag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
