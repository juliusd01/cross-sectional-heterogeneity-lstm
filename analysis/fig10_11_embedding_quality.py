"""Figures 10 and 11: does the learned sector bias line up with realised returns?

Figure 10 plots the yearly weighted correlation between the sector contribution
dC_c and the realised median sector return.  Figure 11 relates that correlation
to how much the sector LSTM beats the base LSTM in the same year, in accuracy
and in Sharpe ratio, with a one-sided permutation test and a Holm correction
across the two hypotheses.

Inputs : tables/embedding_correlation_by_year.csv (tab04_05_sector_contribution.py --all-years)
         which already carries the per-year Sector-minus-Base differences taken
         from tables/annual_metrics_summary.csv (fig09_annual_metrics.py).
Outputs: tables/embedding_quality_stats.csv,
         figures/fig10_embedding_corr_by_year.png, figures/fig11_industry_momentum.png
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from paths import FIGURES, TABLES


def load_data(corr_summary_csv, crisis_years=None):
    if crisis_years is None:
        crisis_years = {2001, 2002, 2008, 2009, 2020}

    df = pd.read_csv(corr_summary_csv)
    expected = {
        "Year": "year",
        "Corr_deltaC_s_Median_Return": "corr",
        "Accuracy_Diff_Sector_minus_Base": "acc",
        "Sharpe_Ratio_Diff_Sector_minus_Base": "sr",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {corr_summary_csv}: {missing}")

    df = df.rename(columns=expected)[["year", "corr", "acc", "sr"]].copy()
    df["crisis"] = df["year"].isin(crisis_years)
    df = df.sort_values("year").reset_index(drop=True)
    return df
 
# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_NORMAL = "#185FA5"
COLOR_CRISIS = "#A32D2D"
COLOR_TREND  = "#888780"
 
def point_color(row):
    if row["crisis"]:
        return COLOR_CRISIS
    return COLOR_NORMAL
 

def permutation_pearson_pvalue(x, y, n_permutations=20000, seed=42):
    rng = np.random.default_rng(seed)
    observed = np.corrcoef(x, y)[0, 1]
    count = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        stat = np.corrcoef(x, y_perm)[0, 1]
        if stat >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def holm_adjust(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        p_adj = (m - rank + 1) * p_values[idx]
        running_max = max(running_max, p_adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted
 
# ── Regression helper ─────────────────────────────────────────────────────────
def add_trendline(ax, x, y):
    slope, intercept, *_ = stats.linregress(x, y)
    xline = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    ax.plot(xline, slope * xline + intercept, color=COLOR_TREND,
            linewidth=1.2, linestyle="--", zorder=1)
 
def scatter_panel(ax, x, y, ylabel, title_text, colors):
    ax.scatter(x, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.4)
    add_trendline(ax, x, y)
    ax.set_title(title_text, fontsize=9, color=COLOR_TREND, pad=4)
    ax.set_xlabel("ΔC$_s$ corr. with realized median return", fontsize=8.5, color=COLOR_TREND)
    ax.set_ylabel(ylabel, fontsize=8.5, color=COLOR_TREND)
    ax.axhline(0, color=COLOR_TREND, linewidth=0.5, linestyle=":")
    ax.axvline(0, color=COLOR_TREND, linewidth=0.5, linestyle=":")
    ax.tick_params(labelsize=8, colors=COLOR_TREND)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")
    ax.set_facecolor("white")
    ax.grid(True, color="#D3D1C7", linewidth=0.4, linestyle="-")
 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--corr-summary-csv",
    default=TABLES / "embedding_correlation_by_year.csv",
    help="Path to yearly correlation summary CSV produced by investigate_sector_weights.py",
)
args = parser.parse_args()

df = load_data(args.corr_summary_csv)
colors = [point_color(row) for _, row in df.iterrows()]

x = df["corr"].to_numpy()
y_acc = df["acc"].to_numpy()
y_sr = df["sr"].to_numpy()

r1, _ = stats.pearsonr(x, y_acc)
r2, _ = stats.pearsonr(x, y_sr)
perm_p = np.array([
    permutation_pearson_pvalue(x, y_acc),
    permutation_pearson_pvalue(x, y_sr),
])
holm_p = holm_adjust(perm_p)

rho1, p_s1 = stats.spearmanr(x, y_acc)
rho2, p_s2 = stats.spearmanr(x, y_sr)

stats_df = pd.DataFrame([
    {"Metric": "Accuracy", "Pearson_r": r1, "Permutation_p": perm_p[0], "Holm_p": holm_p[0], "Spearman_rho": rho1, "Spearman_p": p_s1},
    {"Metric": "Sharpe", "Pearson_r": r2, "Permutation_p": perm_p[1], "Holm_p": holm_p[1], "Spearman_rho": rho2, "Spearman_p": p_s2},
])
stats_df.to_csv(TABLES / "embedding_quality_stats.csv", index=False)
print(stats_df.to_string(index=False))

# ── Figure 1: accuracy + sharpe ───────────────────────────────────────────────
fig1 = plt.figure(figsize=(16, 5.5))
fig1.patch.set_facecolor("white")
gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.28)

ax1 = fig1.add_subplot(gs1[0, 0])
scatter_panel(
    ax1,
    df["corr"],
    df["acc"],
    "Accuracy Difference (Percentage Points)",
    f"r = {r1:.2f}, p_perm = {perm_p[0]:.3f}, p_holm = {holm_p[0]:.3f}",
    colors,
)
 
ax2 = fig1.add_subplot(gs1[0, 1])
scatter_panel(
    ax2,
    df["corr"],
    df["sr"],
    "Sharpe Ratio Difference",
    f"r = {r2:.2f}, p_perm = {perm_p[1]:.3f}, p_holm = {holm_p[1]:.3f}",
    colors,
)
 
from matplotlib.lines import Line2D
legend_elements_scatter = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NORMAL,
           markersize=8, label="Normal year"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CRISIS,
           markersize=8, label="Crisis year"),
    Line2D([0], [0], color=COLOR_TREND, linewidth=1.2, linestyle="--",
           label="OLS trend line"),
]
fig1.legend(handles=legend_elements_scatter, loc="lower center", ncol=4,
            fontsize=8.5, frameon=False,
            bbox_to_anchor=(0.5, -0.04))
fig1.suptitle(
    "Sector embedding quality vs. Sector LSTM outperformance over Base LSTM (1998–2024)",
    fontsize=11, fontweight="500", y=1.02, color="#2C2C2A"
)
fig1.savefig(FIGURES / "fig11_industry_momentum.png", bbox_inches="tight", dpi=300)

# ── Figure 2: yearly ΔCs correlation ──────────────────────────────────────────
fig2, ax4 = plt.subplots(figsize=(12, 4.8))
fig2.patch.set_facecolor("white")
bar_colors = [
    COLOR_NORMAL if row["corr"] >= 0 else COLOR_TREND
    for _, row in df.iterrows()
]
years = df["year"].tolist()
corrs = df["corr"].tolist()
bars = ax4.bar(range(len(years)), corrs, color=bar_colors, width=0.75, zorder=3)

# Crisis markers on the time axis
year_to_idx = {year: idx for idx, year in enumerate(years)}
crisis_events = [
    (2001, "Dot-com crash\n(mid-2000)"),
    (2009, "Financial crisis\n(late-2008)"),
    (2020, "COVID shock\n(early-2020)"),
]
y_min, y_max = ax4.get_ylim()
for event_year, label in crisis_events:
    if event_year in year_to_idx:
        x_pos = year_to_idx[event_year] - 0.5
        ax4.axvline(x=x_pos, color=COLOR_CRISIS, linestyle="--", linewidth=1.0, alpha=0.9, zorder=2)
        ax4.text(
            x_pos + 0.05,
            y_max * 0.97 if y_max > 0 else y_max - 0.03 * (y_max - y_min),
            label,
            color=COLOR_CRISIS,
            fontsize=7.3,
            rotation=90,
            va="top",
            ha="left",
            zorder=4,
        )

ax4.set_xticks(range(len(years)))
ax4.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7.5,
                    color=COLOR_TREND)
ax4.axhline(0, color=COLOR_TREND, linewidth=0.7)
ax4.set_xlabel("Year", fontsize=8.5, color=COLOR_TREND)
ax4.set_ylabel("Corr. with median sector return", fontsize=8.5, color=COLOR_TREND)
ax4.set_title("Embedding quality by year", fontsize=9, color=COLOR_TREND, pad=4)
ax4.tick_params(axis="y", labelsize=8, colors=COLOR_TREND)
for spine in ax4.spines.values():
    spine.set_edgecolor("#D3D1C7")
ax4.set_facecolor("white")
ax4.grid(True, axis="y", color="#D3D1C7", linewidth=0.4, linestyle="-")

fig2.savefig(FIGURES / "fig10_embedding_corr_by_year.png", bbox_inches="tight", dpi=300)

print(f"wrote {FIGURES / 'fig10_embedding_corr_by_year.png'}")
print(f"wrote {FIGURES / 'fig11_industry_momentum.png'}")
