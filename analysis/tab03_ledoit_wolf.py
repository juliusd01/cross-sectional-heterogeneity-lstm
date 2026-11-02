"""
Ledoit-Wolf (2008) studentized circular block bootstrap test for Sharpe ratios.

Reference: Ledoit, O. and Wolf, M. (2008). "Robust Performance Hypothesis Testing
with the Sharpe Ratio." Journal of Empirical Finance, 15(5), 850-859.

Table 3 of the paper.  Two outputs:
  1. Individual SR test  -- H0: SR = 0  vs  H1: SR > 0  (one-sided)
  2. Pairwise SR test    -- H0: SR_i = SR_j  vs  H1: SR_i > SR_j  (one-sided)

Writes tables/tab03_ledoit_wolf_individual.csv,
       tables/tab03_ledoit_wolf_pairwise.csv and tables/tab03_ledoit_wolf.xlsx
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import permutations

from lstm_sector.metrics import load
from lstm_sector.paths import MARKET_RETURN, TABLES, TRADING_DAYS, spec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ANNUAL = TRADING_DAYS   # trading days per year
B_BOOT = 4999             # bootstrap replications
SEED = 42
TRADING_COST = 0.02       # percent per half turn

MODEL_SPECS = {
    "LSTM Base": "base_l3",
    "LSTM Sector": "sector_l3_d2",
    "Random Forest": "rf_n500",
    "LSTM Cov": "cov_l1",
}
MARKET_FILE = MARKET_RETURN


def model_metrics(key: str) -> pd.DataFrame:
    """Daily k=10 long-short returns after transaction costs."""
    return load(spec(key), k=10, trading_cost_pct=TRADING_COST)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def annualized_sharpe(excess_returns: np.ndarray, n: int = N_ANNUAL) -> float:
    mu = excess_returns.mean()
    sigma = excess_returns.std(ddof=1)
    return float(np.sqrt(n) * mu / sigma) if sigma > 1e-12 else np.nan


def lo_se(excess_returns: np.ndarray, n: int = N_ANNUAL) -> float:
    """
    Asymptotic SE of the annualized SR using Lo (2002) / Ledoit-Wolf delta-method.

    Var[sqrt(T) * (SR_daily_hat - SR_daily)] ~= 1 + SR_d^2/2 * k4 - SR_d * k3
    where k3 = skewness, k4 = excess kurtosis of the excess-return series.
    """
    r = excess_returns
    T = len(r)
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma < 1e-12:
        return np.nan
    sr_d = mu / sigma
    k3 = float(((r - mu) ** 3).mean() / sigma ** 3)
    k4 = float(((r - mu) ** 4).mean() / sigma ** 4) - 3.0  # excess kurtosis
    V = 1.0 + sr_d ** 2 * k4 / 2.0 - sr_d * k3
    V = max(V, 1e-10)
    # SE of annualized SR = sqrt(n) * sqrt(V / T)
    return float(np.sqrt(n * V / T))


def circular_block_indices(T: int, L: int, n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([(s + np.arange(L)) % T for s in starts])
    return idx[:T]


# ---------------------------------------------------------------------------
# Individual SR test  (H0: SR = 0)
# ---------------------------------------------------------------------------

def lw_sr_test(
    excess_returns: np.ndarray,
    n_annual: int = N_ANNUAL,
    B: int = B_BOOT,
    L: int | None = None,
    seed: int | None = SEED,
) -> dict:
    """
    Ledoit-Wolf (2008) studentized circular block bootstrap test.
    H0: SR <= 0  vs  H1: SR > 0  (one-sided right).

    Algorithm:
      1. Compute SR_hat and SE_hat (Lo formula).
      2. Center the data under H0: r0 = r - mean(r).
      3. For each bootstrap b:
           draw circular blocks from r0, compute SR*_b and SE*_b,
           form the studentized pivot: t*_b = (SR*_b - SR_hat) / SE*_b.
      4. p-value = #{t*_b >= t_obs} / B  where t_obs = SR_hat / SE_hat.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(excess_returns, dtype=float)
    T = len(r)
    if L is None:
        L = max(1, int(round(T ** (1 / 3))))
    n_blocks = int(np.ceil(T / L))

    sr_hat = annualized_sharpe(r, n=n_annual)
    se_hat = lo_se(r, n=n_annual)
    t_obs = sr_hat / se_hat if (np.isfinite(se_hat) and se_hat > 1e-12) else np.nan

    if not np.isfinite(t_obs):
        return {"sharpe_ratio": sr_hat, "se": se_hat, "t_stat": np.nan, "p_value": np.nan,
                "block_length": L, "n_obs": T}

    # Center under H0
    r0 = r - r.mean()

    t_boot = np.empty(B)
    for b in range(B):
        idx = circular_block_indices(T, L, n_blocks, rng)
        rb = r0[idx]
        sr_b = annualized_sharpe(rb, n=n_annual)
        se_b = lo_se(rb, n=n_annual)
        if not np.isfinite(se_b) or se_b < 1e-12:
            t_boot[b] = 0.0
            continue
        # Studentized pivot: (SR*_b - SR_hat) / SE*_b
        t_boot[b] = (sr_b - sr_hat) / se_b

    p_value = float(np.mean(t_boot >= t_obs))
    return {
        "sharpe_ratio": sr_hat,
        "se": se_hat,
        "t_stat": t_obs,
        "p_value": p_value,
        "block_length": L,
        "n_obs": T,
    }


# ---------------------------------------------------------------------------
# Pairwise SR test  (H0: SR_i = SR_j)
# ---------------------------------------------------------------------------

def lw_sr_pairwise(
    excess_i: np.ndarray,
    excess_j: np.ndarray,
    n_annual: int = N_ANNUAL,
    B: int = B_BOOT,
    L: int | None = None,
    seed: int | None = SEED,
) -> dict:
    """
    Pairwise Ledoit-Wolf test.
    H0: SR_i <= SR_j  vs  H1: SR_i > SR_j  (one-sided right).

    Algorithm:
      1. Compute d_hat = SR_i - SR_j.
      2. Center both series under H0:
           r_i0 = r_i - mean(r_i),  r_j0 = r_j - mean(r_j).
      3. Bootstrap jointly (same block indices preserve cross-correlation):
           d*_b = SR*(r_ib) - SR*(r_jb).
      4. SE_diff = std(d*_b)  [bootstrap SE; naturally accounts for cross-correlation].
         t_obs  = d_hat / SE_diff.
      5. Percentile p-value: p = #{d*_b >= d_hat} / B.
         Uses the same bootstrap SE in the denominator of t_obs and the p-value
         so they are on a consistent scale.
    """
    rng = np.random.default_rng(seed)
    ri = np.asarray(excess_i, dtype=float)
    rj = np.asarray(excess_j, dtype=float)
    T = len(ri)
    assert len(rj) == T, "Series must have the same length after alignment."

    if L is None:
        L = max(1, int(round(T ** (1 / 3))))
    n_blocks = int(np.ceil(T / L))

    sr_i = annualized_sharpe(ri, n=n_annual)
    sr_j = annualized_sharpe(rj, n=n_annual)
    d_hat = sr_i - sr_j

    # Center both series under H0
    ri0 = ri - ri.mean()
    rj0 = rj - rj.mean()

    d_boot = np.empty(B)

    for b in range(B):
        idx = circular_block_indices(T, L, n_blocks, rng)
        rib = ri0[idx]
        rjb = rj0[idx]
        d_boot[b] = annualized_sharpe(rib, n=n_annual) - annualized_sharpe(rjb, n=n_annual)

    # Bootstrap SE accounts for cross-series correlation via joint resampling
    se_diff = float(d_boot.std())
    if se_diff < 1e-12:
        return {"sr_i": sr_i, "sr_j": sr_j, "sr_diff": d_hat,
                "se_diff": se_diff, "t_stat": np.nan, "p_value": np.nan,
                "block_length": L, "n_obs": T}

    t_obs = d_hat / se_diff
    # Percentile bootstrap: fraction of bootstrap SR diffs >= observed SR diff
    p_value = float(np.mean(d_boot >= d_hat))
    return {
        "sr_i": sr_i,
        "sr_j": sr_j,
        "sr_diff": d_hat,
        "se_diff": se_diff,
        "t_stat": t_obs,
        "p_value": p_value,
        "block_length": L,
        "n_obs": T,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_excess_returns(model_name: str, key: str, rf: pd.Series) -> pd.Series:
    df = model_metrics(key)
    df = df.merge(rf.rename("rf"), left_on="date", right_index=True, how="inner")
    excess = (df["avg_daily_return"] - df["rf"]).values
    return pd.Series(excess, index=df["date"], name=model_name)


def load_market_excess_returns(path: str, start_date: str, end_date: str) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    # mkt-rf is already the market excess return in French data
    return pd.Series(df["mkt-rf"].values, index=df["date"], name="Market")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load data ---
    market_raw = pd.read_csv(MARKET_FILE)
    market_raw["date"] = pd.to_datetime(market_raw["date"], format="%Y%m%d")
    rf = market_raw.set_index("date")["rf"]

    # Load model excess returns aligned to each model's date range
    model_returns: dict[str, np.ndarray] = {}
    model_dates: dict[str, pd.DatetimeIndex] = {}

    for name, key in MODEL_SPECS.items():
        df = model_metrics(key)
        df = df.merge(rf.rename("rf"), left_on="date", right_index=True, how="inner")
        excess = (df["avg_daily_return"] - df["rf"]).values
        model_returns[name] = excess.astype(float)
        model_dates[name] = df["date"].values

    # Market: use the common date range across all models
    all_dates = set(model_dates[next(iter(MODEL_SPECS))])
    for d in model_dates.values():
        all_dates &= set(d)
    common_dates = sorted(all_dates)

    market_excess = (
        market_raw.set_index("date")
        .loc[lambda x: x.index.isin(common_dates), "mkt-rf"]
        .sort_index()
        .values.astype(float)
    )
    model_returns["Market"] = market_excess

    # Align all model series to the same common dates
    aligned: dict[str, np.ndarray] = {}
    for name, key in MODEL_SPECS.items():
        df = model_metrics(key)
        df = df.merge(rf.rename("rf"), left_on="date", right_index=True, how="inner")
        df = df[df["date"].isin(set(common_dates))].sort_values("date")
        aligned[name] = (df["avg_daily_return"] - df["rf"]).values.astype(float)
    aligned["Market"] = market_excess

    model_order = ["Market"] + list(MODEL_SPECS.keys())
    T = len(common_dates)
    print(f"Common trading days: {T}  ({common_dates[0]} – {common_dates[-1]})")

    # -----------------------------------------------------------------------
    # 1. Individual SR tests
    # -----------------------------------------------------------------------
    print("\n=== Individual Sharpe Ratio Tests (H0: SR = 0) ===")
    individual_rows = []
    for name in model_order:
        r = aligned[name]
        result = lw_sr_test(r, n_annual=N_ANNUAL, B=B_BOOT, seed=SEED)
        row = {
            "Model": name,
            "Sharpe Ratio": round(result["sharpe_ratio"], 4),
            "SE": round(result["se"], 4),
            "t-stat": round(result["t_stat"], 4),
            "p-value": round(result["p_value"], 4),
            "Sig. 10%": "*" if result["p_value"] < 0.10 else "",
            "Sig. 5%": "**" if result["p_value"] < 0.05 else "",
            "Sig. 1%": "***" if result["p_value"] < 0.01 else "",
            "Block Length": result["block_length"],
            "N": result["n_obs"],
        }
        individual_rows.append(row)
        print(f"  {name:20s}  SR={row['Sharpe Ratio']:6.3f}  t={row['t-stat']:6.3f}  p={row['p-value']:.4f}")

    individual_df = pd.DataFrame(individual_rows)

    # -----------------------------------------------------------------------
    # 2. Pairwise SR tests (all ordered pairs: H0: SR_i <= SR_j)
    # -----------------------------------------------------------------------
    print("\n=== Pairwise Sharpe Ratio Tests (H0: SR_i <= SR_j) ===")
    pairwise_rows = []
    for i, j in permutations(model_order, 2):
        result = lw_sr_pairwise(aligned[i], aligned[j], n_annual=N_ANNUAL, B=B_BOOT, seed=SEED)
        row = {
            "Model i": i,
            "Model j": j,
            "SR_i": round(result["sr_i"], 4),
            "SR_j": round(result["sr_j"], 4),
            "SR_i - SR_j": round(result["sr_diff"], 4),
            "SE(diff)": round(result["se_diff"], 4),
            "t-stat": round(result["t_stat"], 4),
            "p-value": round(result["p_value"], 4),
            "Sig. 10%": "*" if result["p_value"] < 0.10 else "",
            "Sig. 5%": "**" if result["p_value"] < 0.05 else "",
            "Sig. 1%": "***" if result["p_value"] < 0.01 else "",
            "Block Length": result["block_length"],
            "N": result["n_obs"],
        }
        pairwise_rows.append(row)
        print(f"  {i:20s} vs {j:20s}  diff={row['SR_i - SR_j']:+.3f}  t={row['t-stat']:+.3f}  p={row['p-value']:.4f}")

    pairwise_df = pd.DataFrame(pairwise_rows)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_individual = TABLES / "tab03_ledoit_wolf_individual.csv"
    out_pairwise = TABLES / "tab03_ledoit_wolf_pairwise.csv"
    out_excel = TABLES / "tab03_ledoit_wolf.xlsx"

    individual_df.to_csv(out_individual, index=False)
    pairwise_df.to_csv(out_pairwise, index=False)

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        individual_df.to_excel(writer, sheet_name="Individual SR Test", index=False)
        pairwise_df.to_excel(writer, sheet_name="Pairwise SR Test", index=False)

    print(f"\nSaved: {out_individual}")
    print(f"Saved: {out_pairwise}")
    print(f"Saved: {out_excel}")
    print(f"\nBootstrap settings: B={B_BOOT}, L=floor(T^(1/3))={int(round(T**(1/3)))}, seed={SEED}")


if __name__ == "__main__":
    main()
