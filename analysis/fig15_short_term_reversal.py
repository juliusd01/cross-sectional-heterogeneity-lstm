#!/usr/bin/env python3
"""Figure 15: short-term reversal in the temporal signal.

Averages the 60 trading days of returns that precede each rebalancing date,
separately for the ten stocks picked long, the ten picked short, and the whole
S&P 500 universe.  The long picks are recent losers and the short picks recent
winners -- the reversal signature of Jegadeesh (1990) and Lehmann (1990),
recovered here from the LSTM's temporal component alone.

Needs the raw price panels under data/prices/ -- see data/README.md.

Writes tables/short_term_reversal_summary.csv and
       figures/fig15_short_term_reversal.png
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from metrics import load
from paths import FIGURES, PRICES, TABLES, spec

LOOKBACK = 60


def load_all_test_returns(input_dir: Path) -> pd.DataFrame:
    test_files = sorted(input_dir.glob("test_*.csv"))
    if not test_files:
        raise SystemExit(
            f"no test_*.csv files in {input_dir}.  The Refinitiv price panels are\n"
            "not redistributable; see data/README.md for the required format."
        )

    yearly_returns = []
    for file_path in test_files:
        data = pd.read_csv(file_path)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")

        price_data = data.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")
        rets = price_data.pct_change(fill_method=None)
        rets.insert(0, "Date", data["Date"].values)
        rets = rets.iloc[1:].copy()
        yearly_returns.append(rets)

    returns_df = pd.concat(yearly_returns, axis=0, ignore_index=True)
    returns_df = returns_df.dropna(subset=["Date"]).sort_values("Date")
    returns_df = returns_df.drop_duplicates(subset=["Date"], keep="first")
    returns_df = returns_df.set_index("Date")
    returns_df = returns_df.sort_index()
    return returns_df


def safe_parse_list(raw_value: str) -> list[str]:
    try:
        value = ast.literal_eval(raw_value)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, str)]
    except (ValueError, SyntaxError):
        return []
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="sector_l3_d2")
    parser.add_argument("--trading-cost", type=float, default=0.02)
    parser.add_argument("--write-sequences", action="store_true",
                        help="also dump the per-rebalance-date sequences "
                             "(a ~40 MB intermediate, not needed for the figure)")
    args = parser.parse_args()

    print("Loading metrics...")
    metrics = load(spec(args.model), k=10,
                   trading_cost_pct=args.trading_cost, lists=True)
    metrics = metrics.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print("Loading and merging test return data...")
    returns = load_all_test_returns(PRICES)
    date_index = pd.Index(returns.index)

    print(f"Merged returns shape: {returns.shape}")
    print(f"Metrics rows: {len(metrics)}")

    records = []
    skipped_no_date = 0
    skipped_short_history = 0

    for _, row in metrics.iterrows():
        rebalance_date = row["date"]
        if rebalance_date not in date_index:
            skipped_no_date += 1
            continue

        loc = date_index.get_loc(rebalance_date)
        if isinstance(loc, slice):
            loc = loc.start
        elif not isinstance(loc, int):
            loc = int(loc[0])

        if loc < LOOKBACK:
            skipped_short_history += 1
            continue

        window = returns.iloc[loc - LOOKBACK : loc]  # previous 60 trading days
        if len(window) < LOOKBACK:
            skipped_short_history += 1
            continue

        top_tickers = [t for t in row["top_stocks"] if t in returns.columns]
        flop_tickers = [t for t in row["flop_stocks"] if t in returns.columns]

        seq_top = window[top_tickers].mean(axis=1, skipna=True) if top_tickers else pd.Series(index=window.index, dtype=float)
        seq_flop = window[flop_tickers].mean(axis=1, skipna=True) if flop_tickers else pd.Series(index=window.index, dtype=float)
        seq_all = window.mean(axis=1, skipna=True)

        for i, dt in enumerate(window.index):
            lag = -(LOOKBACK - i)  # -60, ..., -1
            records.append(
                {
                    "rebalance_date": rebalance_date,
                    "lookback_date": dt,
                    "lag": lag,
                    "top_mean_return": float(seq_top.loc[dt]) if not seq_top.empty else float("nan"),
                    "flop_mean_return": float(seq_flop.loc[dt]) if not seq_flop.empty else float("nan"),
                    "all_mean_return": float(seq_all.loc[dt]),
                    "n_top_stocks": len(top_tickers),
                    "n_flop_stocks": len(flop_tickers),
                }
            )

    seq_df = pd.DataFrame(records)
    if seq_df.empty:
        raise RuntimeError("No lookback sequences were generated. Check date overlap between metrics and test files.")

    sequence_file = TABLES / "short_term_reversal_sequences.csv"
    if args.write_sequences:
        seq_df.to_csv(sequence_file, index=False)

    long_df = seq_df.melt(
        id_vars=["rebalance_date", "lookback_date", "lag", "n_top_stocks", "n_flop_stocks"],
        value_vars=["top_mean_return", "flop_mean_return", "all_mean_return"],
        var_name="portfolio_leg",
        value_name="mean_return",
    )
    long_df["portfolio_leg"] = long_df["portfolio_leg"].map(
        {
            "top_mean_return": "Top (Long)",
            "flop_mean_return": "Flop (Short candidates)",
            "all_mean_return": "All Stocks (Universe Mean)",
        }
    )

    summary = (
        long_df.groupby(["portfolio_leg", "lag"], as_index=False)
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("mean_return", "std"),
            n_obs=("mean_return", "count"),
        )
        .sort_values(["portfolio_leg", "lag"])
    )

    # Build cumulative path from average daily return at each lag.
    summary["cum_path"] = (
        summary.sort_values("lag")
        .groupby("portfolio_leg")["mean_return"]
        .transform(lambda x: (1 + x.fillna(0)).cumprod() - 1)
    )

    summary_file = TABLES / "short_term_reversal_summary.csv"
    summary.to_csv(summary_file, index=False)

    # Plot: cumulative path only (publication-ready, no title).
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    colors = {
        "Top (Long)": "#2ca02c",
        "Flop (Short candidates)": "#d62728",
        "All Stocks (Universe Mean)": "#1f77b4",
    }

    for leg, grp in summary.groupby("portfolio_leg"):
        grp = grp.sort_values("lag")
        ax.plot(grp["lag"], grp["cum_path"], label=leg, color=colors.get(leg))

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative return")
    ax.set_xlim(-LOOKBACK, 0)
    ax.set_xticks([-60, -40, -20, 0])
    ax.set_xticklabels(["t-60", "t-40", "t-20", "t"])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plot_file = FIGURES / "fig15_short_term_reversal.png"
    fig.savefig(plot_file, dpi=300)

    print("\n=== Done ===")
    if args.write_sequences:
        print(f"Sequences file: {sequence_file}")
    print(f"Summary file:   {summary_file}")
    print(f"Plot file:      {plot_file}")
    print(f"Generated rows: {len(seq_df)}")
    print(f"Skipped (date missing): {skipped_no_date}")
    print(f"Skipped (insufficient lookback): {skipped_short_history}")


if __name__ == "__main__":
    main()
