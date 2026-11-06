"""Table 6: robustness of the long-short strategy across model specifications.

Performance of every trained model version (mainly varying the number of LSTM
layers), pooled across the 27 study periods (1998-2024).

Metrics per version (k = 10 long-short portfolio, daily rebalancing):
  - accuracy of the k=10 portfolio (share of correctly classified selected stocks)
  - full-sample classification accuracy over all stocks (from the prediction files)
  - mean daily return in % (before and, where available, after 2bps costs)
  - annualized Sharpe ratio (excess over the risk-free rate), pooled over all days
  - mean +/- sd of the 27 annual Sharpe ratios, and the number of years with a
    positive mean daily return

Outputs:
  tables/tab06_robustness.csv            (all versions, all metrics)
  tables/tab06_robustness_lstm.tex       (panel A: LSTM architectures)
  tables/tab06_robustness_benchmark.tex  (panel B: tree-based benchmarks)
  tables/robustness_annual_metrics.csv   (per-version, per-period detail)

Acc_all needs the raw prediction files.  They are not committed, so the values
are read from the metrics/accuracy_all.csv cache when predictions/ is absent.
"""

import argparse
import glob

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from metrics import load, risk_free_rate
from paths import (LSTM_FAMILIES, METRICS, SPECS, TABLES,
                               TRADING_DAYS)

OUT = TABLES



def newey_west_t(series, lags=1):
    """Newey-West (1 lag) t-statistic for the mean of a daily return series."""
    series = np.asarray(series, dtype=float)
    model = OLS(series, np.ones(len(series))).fit(cov_type='HAC',
                                                  cov_kwds={'maxlags': lags})
    return model.tvalues[0]


def pooled_stats(df, rf, prefix=''):
    """Pooled full-sample statistics over all daily observations."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(rf, left_on='date', right_index=True, how='left')

    r = df['avg_daily_return']
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    excess = ann_ret - df['rf'].mean() * TRADING_DAYS

    return {
        f'{prefix}mean_daily_return': r.mean(),
        f'{prefix}t_stat_NW': newey_west_t(r),
        f'{prefix}std_daily_return': r.std(),
        f'{prefix}annualized_return': ann_ret / 100,
        f'{prefix}annualized_volatility': ann_vol,
        f'{prefix}sharpe_ratio': excess / ann_vol,
    }


def annual_stats(df, rf):
    """Per-study-period statistics and their distribution across the 27 periods."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(rf, left_on='date', right_index=True, how='left')

    per_year = df.groupby('year').apply(
        lambda g: pd.Series({
            'mean_return': g['avg_daily_return'].mean(),
            'sharpe': ((g['avg_daily_return'].mean() - g['rf'].mean()) * TRADING_DAYS)
                      / (g['avg_daily_return'].std() * np.sqrt(TRADING_DAYS)),
            # the stored 'accuracy' column is rounded to two decimals, so the
            # portfolio accuracy is rebuilt from the two unrounded legs
            'accuracy': (g['accuracy_top_k'] + g['accuracy_flop_k']).mean() / 2,
        }), include_groups=False)

    return {
        'n_periods': len(per_year),
        'sharpe_mean_across_periods': per_year['sharpe'].mean(),
        'sharpe_sd_across_periods': per_year['sharpe'].std(),
        'sharpe_min_across_periods': per_year['sharpe'].min(),
        'periods_positive_return': int((per_year['mean_return'] > 0).sum()),
        'periods_accuracy_above_50': int((per_year['accuracy'] > 0.5).sum()),
    }, per_year


def accuracy_cache():
    """Per-spec, per-year cross-sectional accuracy, precomputed from predictions/."""
    path = METRICS / 'accuracy_all.csv'
    return pd.read_csv(path) if path.exists() else None


def full_sample_accuracy(spec, cache):
    """Classification accuracy over *all* stocks and days, pooled across periods."""
    files = sorted(glob.glob(str(spec.predictions / 'predictions_*.parquet')))
    if files:
        correct = total = 0
        for f in files:
            d = pd.read_parquet(f, columns=['predicted_class', 'actual_class'])
            correct += int((d['predicted_class'] == d['actual_class']).sum())
            total += len(d)
        return correct / total if total else np.nan

    if cache is None:
        return np.nan
    rows = cache[cache['spec'] == spec.key]
    if rows.empty:
        return np.nan
    # weight each period by its number of observations
    return float((rows['accuracy_all'] * rows['n_obs']).sum() / rows['n_obs'].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.parse_args()

    rf = risk_free_rate()
    cache = accuracy_cache()
    rows = []
    annual_rows = []

    for spec in SPECS:
        print(f'processing {spec.path} ...', flush=True)

        metrics = load(spec, k=10)
        row = {
            'family': spec.family,
            'version': spec.path.split('/')[1],
            'path': spec.path,
            'layers': spec.layers,
            'note': spec.note,
            'main_specification': spec.main,
            'portfolio_accuracy_k10': (metrics['accuracy_top_k']
                                       + metrics['accuracy_flop_k']).mean() / 2,
            'accuracy_long_leg': metrics['accuracy_top_k'].mean(),
            'accuracy_short_leg': metrics['accuracy_flop_k'].mean(),
            'full_sample_accuracy': full_sample_accuracy(spec, cache),
        }
        row.update(pooled_stats(metrics, rf))

        ann, per_year = annual_stats(metrics, rf)
        row.update(ann)

        per_year = per_year.reset_index()
        per_year.insert(0, 'path', spec.path)
        annual_rows.append(per_year)

        row.update(pooled_stats(load(spec, k=10, trading_cost_pct=0.02),
                                rf, prefix='tc2bps_'))

        rows.append(row)

    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / 'tab06_robustness.csv', index=False)
    pd.concat(annual_rows, ignore_index=True).to_csv(
        OUT / 'robustness_annual_metrics.csv', index=False)
    print(f'\nwrote {OUT / "tab06_robustness.csv"}')

    in_table = table[table['path'].isin(
        [spec.path for spec in SPECS if spec.in_table])]

    write_latex(in_table[in_table['family'].isin(LSTM_FAMILIES)],
                OUT / 'tab06_robustness_lstm.tex',
                caption=('Robustness of the long-short strategy ($k=10$) across LSTM '
                         'architectures. All figures are pooled over the 27 study '
                         'periods (1998--2024).'),
                label='tab:robustness_lstm', show_layers=True)

    write_latex(in_table[~in_table['family'].isin(LSTM_FAMILIES)],
                OUT / 'tab06_robustness_benchmark.tex',
                caption=('Robustness of the long-short strategy ($k=10$) across the '
                         'tree-based benchmark specifications. All figures are pooled '
                         'over the 27 study periods (1998--2024).'),
                label='tab:robustness_benchmark', show_layers=False)

    # console preview
    show = ['family', 'version', 'layers', 'portfolio_accuracy_k10',
            'full_sample_accuracy', 'mean_daily_return', 'sharpe_ratio',
            'tc2bps_mean_daily_return', 'tc2bps_sharpe_ratio',
            'sharpe_mean_across_periods', 'sharpe_sd_across_periods',
            'periods_positive_return']
    with pd.option_context('display.width', 220, 'display.max_columns', None):
        print()
        print(table[show].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


def write_latex(sub, path, caption, label, show_layers):
    """booktabs table, one row per model version, grouped by family."""
    tail_header = (r'Acc.$_{k=10}$ & Acc.$_{all}$ & $\bar{r}$ & $t_{NW}$ & SR & '
                   r'$\bar{r}^{tc}$ & SR$^{tc}$ & $\overline{SR}_{y}$ (sd) \\')
    if show_layers:
        colspec = 'll' + 'c' * 8
        header = rf'Layers & Hyper-parameters & {tail_header}'
        n_cols = 10
    else:
        colspec = 'l' + 'c' * 8
        header = rf'Specification & {tail_header}'
        n_cols = 9

    def fmt(v, dec=3):
        return '--' if pd.isna(v) else f'{v:.{dec}f}'

    lines = [r'\begin{table}[htbp]', r'\centering',
             rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\small',
             rf'\begin{{tabular}}{{{colspec}}}', r'\toprule',
             header, r'\midrule']

    for family, grp in sub.groupby('family', sort=False):
        lines.append(rf'\multicolumn{{{n_cols}}}{{l}}{{\textit{{{family}}}}} \\')
        for _, r in grp.iterrows():
            first = str(int(r['layers'])) if show_layers and pd.notna(r['layers']) else ''
            second = r['note'] if r['note'] else '--'
            star = r'$^{\dagger}$' if r['main_specification'] else ''
            label_cells = (f'{first}{star} & {second}' if show_layers
                           else f'{second}{star}')
            sr_y = f"{r['sharpe_mean_across_periods']:.2f} ({r['sharpe_sd_across_periods']:.2f})"
            lines.append(
                fr'\quad {label_cells} & {fmt(r["portfolio_accuracy_k10"])} & '
                fr'{fmt(r["full_sample_accuracy"])} & {fmt(r["mean_daily_return"], 3)} & '
                fr'{fmt(r["t_stat_NW"], 2)} & {fmt(r["sharpe_ratio"], 2)} & '
                fr'{fmt(r["tc2bps_mean_daily_return"], 3)} & '
                fr'{fmt(r["tc2bps_sharpe_ratio"], 2)} & {sr_y} \\')
        lines.append(r'\addlinespace')

    if lines[-1] == r'\addlinespace':
        lines.pop()

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{minipage}{\textwidth}\vspace{2mm}\footnotesize',
        r'Notes: Acc.$_{k=10}$ is the directional accuracy of the 20 stocks held each day '
        r'(10 long, 10 short); Acc.$_{all}$ is the classification accuracy over all stocks '
        r'in the cross-section. $\bar{r}$ is the mean daily return of the long-short '
        r'portfolio in percent, $t_{NW}$ its Newey-West (1 lag) $t$-statistic, and SR the '
        r'annualized Sharpe ratio computed on excess returns over the risk-free rate. '
        r'Superscript $tc$ denotes figures after transaction costs of 2 bps per trade. '
        r'$\overline{SR}_{y}$ is the mean of the '
        r'27 period-specific annualized Sharpe ratios, with the standard deviation across '
        r'periods in parentheses. $^{\dagger}$ marks the specification reported in the '
        r'main results.',
        r'\end{minipage}',
        r'\end{table}', '']

    path.write_text('\n'.join(lines))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
