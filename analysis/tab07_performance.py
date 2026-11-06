"""Table 7 (appendix): risk and return characteristics of the long-short portfolios.

Pools all daily observations from 1998-2024 for the sector LSTM, the base LSTM
and the random forest -- each after 2 bps of transaction costs per half turn --
and for the value-weighted market benchmark.  Figures 7 and 8 read the table
this script writes.

Writes tables/tab07_return_characteristics.csv and tables/tab07_risk_characteristics.xlsx
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from metrics import load, market_return, risk_free_rate
from paths import DATA, TABLES, TRADING_DAYS, spec

# How the strategy return series is scaled before its characteristics are
# computed.  Only the strategies are rescaled; the market is always in percent.
#   percent   raw percentage returns (the correct default)
#   volmatch  levered to the market's annualised volatility, so tail and
#             drawdown measures are compared at equal risk budget
#   sigma     divided by each period's training standard deviation -- what the
#             original run effectively reported.  Kept for comparison only; the
#             resulting numbers are not in any interpretable unit.
SCALES = ('percent', 'volmatch', 'sigma')

# Model column order in the table, and the spec each one refers to.
MODELS = {
    'LSTM Base': 'base_l3',
    'LSTM Sector': 'sector_l3_d2',
    'Random Forest': 'rf_n500',
}

RISK_METRICS = [
    'std_dev', 'annualized_volatility', 'downside_deviation',
    'min_return', 'max_return',
    'skewness', 'kurtosis', 'one_percent_VaR', 'five_percent_VaR',
    'one_percent_CVaR', 'five_percent_CVaR', 'max_drawdown',
]


def newey_west_t(series: pd.Series, lags: int = 1) -> float:
    """t-statistic for the mean of a daily return series, HAC with one lag."""
    model = OLS(series, np.ones(len(series))).fit(
        cov_type='HAC', cov_kwds={'maxlags': lags})
    return model.tvalues.iloc[0]


def characteristics(returns: pd.Series, rf: pd.Series,
                    long_leg: pd.Series | float = 0.0,
                    short_leg: pd.Series | float = 0.0) -> dict:
    """Return and risk characteristics of one daily return series."""
    var1 = returns.quantile(0.01)
    var5 = returns.quantile(0.05)

    # Maximum drawdown of the compounded series.
    wealth = (1 + returns / 100).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()

    annualized_return = returns.mean() * TRADING_DAYS
    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    # dispersion of losses only; unlike total volatility it still separates the
    # models once they are levered to a common volatility target
    losses = returns[returns < 0]
    downside_deviation = np.sqrt((losses ** 2).mean()) * np.sqrt(TRADING_DAYS)
    excess = annualized_return - rf.mean() * TRADING_DAYS

    return {
        'mean_return': returns.mean(),
        'mean_return_long': np.mean(long_leg),
        'mean_return_short': np.mean(short_leg),
        'std_dev': returns.std(),
        'min_return': returns.min(),
        'max_return': returns.max(),
        'median_return': returns.median(),
        'first_quartile': returns.quantile(0.25),
        'third_quartile': returns.quantile(0.75),
        'share_positive_returns': (returns > 0).mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        't-statistics': newey_west_t(returns),
        'one_percent_VaR': var1,
        'five_percent_VaR': var5,
        'one_percent_CVaR': returns[returns <= var1].mean(),
        'five_percent_CVaR': returns[returns <= var5].mean(),
        'max_drawdown': drawdown.min(),
        'annualized_return': annualized_return / 100,
        'annualized_volatility': annualized_volatility,
        'downside_deviation': downside_deviation,
        'excess_annualized_return': excess / 100,
        'sharpe_ratio': excess / annualized_volatility,
    }


def rescale(df: pd.DataFrame, scale: str, market_vol: float) -> pd.DataFrame:
    """Scale the three return columns of one strategy."""
    columns = ['avg_daily_return', 'avg_return_top_k', 'avg_return_flop_k']
    out = df.copy()
    if scale == 'percent':
        return out
    if scale == 'volmatch':
        factor = market_vol / (out['avg_daily_return'].std() * np.sqrt(TRADING_DAYS))
    elif scale == 'sigma':
        sigma = pd.read_csv(DATA / 'standardisation_reference.csv')
        factor = 1 / out['year'].map(sigma.set_index('study_period')['sigma_train_pct'])
    else:
        raise SystemExit(f'unknown scale {scale!r}; choose from {SCALES}')
    for column in columns:
        out[column] = out[column] * factor
    return out


def build(trading_cost_pct: float, k: int = 10,
          scale: str = 'volmatch') -> pd.DataFrame:
    rf = risk_free_rate()

    market = market_return()
    market_vol = market['return'].std() * np.sqrt(TRADING_DAYS)
    columns = {'Market': characteristics(market['return'],
                                         market.set_index('date')['rf'])}

    for label, key in MODELS.items():
        df = load(spec(key), k=k, trading_cost_pct=trading_cost_pct)
        df = df.merge(rf, left_on='date', right_index=True, how='left')
        df = rescale(df, scale, market_vol)
        columns[label] = characteristics(
            df['avg_daily_return'], df['rf'],
            df['avg_return_top_k'], df['avg_return_flop_k'])

    return pd.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trading-cost', type=float, default=0.02,
                        help='cost per half turn in percent (default: 0.02 = 2 bps)')
    parser.add_argument('--k', type=int, default=10,
                        help='positions per leg (default: 10, the paper table)')
    parser.add_argument('--scale', choices=SCALES, default='volmatch',
                        help='how to scale the strategy returns (default: volmatch, '
                             'matching Figures 7 and 8)')
    args = parser.parse_args()

    table = build(args.trading_cost, args.k, args.scale)
    # k = 10, vol-matched is Table 7 itself and takes the plain name; other
    # sizes and scalings go to side tables.
    suffix = '' if args.k == 10 else f'_k{args.k}'
    if args.scale != 'volmatch':
        suffix += f'_{args.scale}'
    csv = TABLES / f'tab07_return_characteristics{suffix}.csv'
    xlsx = TABLES / f'tab07_risk_characteristics{suffix}.xlsx'
    table.to_csv(csv)
    table.loc[RISK_METRICS].to_excel(xlsx)

    print(table.round(3).to_string())
    print(f'\nwrote {csv}\nwrote {xlsx}')


if __name__ == '__main__':
    main()
