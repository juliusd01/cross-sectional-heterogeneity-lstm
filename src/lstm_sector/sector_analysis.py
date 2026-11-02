"""Decomposing the sector-embedding signal (section 4.3 of the paper).

The final linear layer maps the concatenation of the LSTM hidden state and the
sector embedding onto the two class logits:

    z = W_h h_T + W_e e_c + b

Because the sector term is a plain additive offset, the contribution of sector
c to the long-vs-short decision is the scalar

    dC_c = (W_e e_c)_1 - (W_e e_c)_0

If dC_c > 0 the embedding tilts stocks in sector c towards the long leg.  This
module extracts dC_c from the saved weights and compares it with the realised
performance of the sector, in the training window and in the test year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .paths import COMPANY_SECTOR, PRICES, SECTORS, SECTORS_11


def load_returns(year: int) -> pd.DataFrame:
    """Daily returns for the whole four-year study period, wide format."""
    path = PRICES / f'test_{year}.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found.  The Refinitiv price panels are not redistributable;\n'
            'see data/README.md for the required format.'
        )
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])
    returns = data.drop(columns='Date').pct_change(fill_method=None)
    returns['Date'] = data['Date']
    return returns.iloc[1:]


def sector_map() -> dict:
    sectors = pd.read_csv(COMPANY_SECTOR, usecols=['Instrument', 'Economic_Sector'])
    return dict(zip(sectors['Instrument'], sectors['Economic_Sector']))


def sector_contribution(state_dict) -> pd.DataFrame:
    """dC_c per sector, read straight off the trained output layer."""
    embedding = state_dict['sector_embedding.weight']
    fc_weights = state_dict['fc.weight']

    # fc sees [hidden_state, sector_embedding]; the embedding block is the tail.
    hidden_size = fc_weights.shape[1] - embedding.shape[1]
    embedding_to_output = fc_weights[:, hidden_size:]

    z0 = (embedding @ embedding_to_output[0]).numpy()
    z1 = (embedding @ embedding_to_output[1]).numpy()

    # The universe occasionally includes Academic & Educational Services, which
    # adds an eleventh embedding row.
    names = SECTORS_11 if embedding.shape[0] == 11 else SECTORS

    df = pd.DataFrame({
        'Sector': names,
        'Contribution_to_z0_SHORT': z0,
        'Contribution_to_z1_LONG': z1,
    })
    df['Difference_(z1-z0)'] = df['Contribution_to_z1_LONG'] - df['Contribution_to_z0_SHORT']
    return df


def sector_performance(returns: pd.DataFrame, stoi: dict, year: int) -> pd.DataFrame:
    """Realised sector performance in the test year and the training window."""
    stock_sectors = {c: stoi[c] for c in returns.columns
                     if c != 'Date' and c in stoi}

    test_mask = returns['Date'].dt.year == year
    train_mask = returns['Date'].dt.year.between(year - 3, year - 1)

    rows = []
    for sector in sorted(set(stock_sectors.values())):
        stocks = [s for s, sec in stock_sectors.items() if sec == sector]
        if not stocks:
            continue
        test = returns.loc[test_mask, stocks]
        train = returns.loc[train_mask, stocks]

        rows.append({
            'Sector': sector,
            'Train_Mean_Return_3Y': train.mean().mean(),
            'Train_Median_Return_3Y': train.median().median(),
            'Train_Total_Return_3Y': (1 + train).prod().median() - 1,
            'Mean_Return': test.mean().mean(),
            'Median_Return': test.median().median(),
            'Pct_Positive_Returns': (test > 0).sum().sum() / test.notna().sum().sum(),
            'Total_Return': (1 + test).prod().median() - 1,
            'Num_Stocks': len(stocks),
        })
    return pd.DataFrame(rows)


def weighted_corr(x, y, w) -> float:
    """Pearson correlation weighted by the number of stocks per sector."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    cov = np.sum(w * (x - mean_x) * (y - mean_y))
    var_x = np.sum(w * (x - mean_x) ** 2)
    var_y = np.sum(w * (y - mean_y) ** 2)
    return float(cov / np.sqrt(var_x * var_y))


def load_state_dict(model_dir, year: int):
    path = model_dir / f'best_lstm_model_{year}.pth'
    if not path.exists():
        raise SystemExit(f'{path} not found')
    return torch.load(path, map_location='cpu', weights_only=True)


def compare(model_dir, year: int) -> pd.DataFrame:
    """dC_c next to realised sector performance, for one study period."""
    contribution = sector_contribution(load_state_dict(model_dir, year))
    performance = sector_performance(load_returns(year), sector_map(), year)
    merged = contribution.merge(performance, on='Sector', how='inner')
    return merged.sort_values('Difference_(z1-z0)', ascending=False)


RETURN_MEASURES = [
    'Train_Mean_Return_3Y', 'Train_Median_Return_3Y', 'Train_Total_Return_3Y',
    'Mean_Return', 'Median_Return', 'Pct_Positive_Returns', 'Total_Return',
]


def correlations(merged: pd.DataFrame) -> dict:
    """Weighted correlation of dC_c with every realised return measure."""
    delta = merged['Difference_(z1-z0)'].values
    weights = merged['Num_Stocks'].values
    return {m: weighted_corr(delta, merged[m].values, weights) for m in RETURN_MEASURES}


def sector_summary_statistics(start_year: int, end_year: int) -> pd.DataFrame:
    """Table 1: yearly equal-weighted sector statistics over the whole sample."""
    stoi = sector_map()
    annual_returns: dict[str, list] = {}
    stock_counts: dict[str, list] = {}

    for year in range(start_year, end_year + 1):
        try:
            returns = load_returns(year)
        except SystemExit:
            print(f'warning: no price file for {year}, skipping')
            continue

        stock_sectors = {c: stoi[c] for c in returns.columns
                         if c != 'Date' and c in stoi}
        in_year = returns['Date'].dt.year == year

        for sector in sorted(set(stock_sectors.values())):
            stocks = [s for s, sec in stock_sectors.items() if sec == sector]
            if not stocks:
                continue
            # equal-weighted sector return: compound each stock, then average
            annual = (1 + returns.loc[in_year, stocks]).prod().mean() - 1
            annual_returns.setdefault(sector, []).append(annual)
            stock_counts.setdefault(sector, []).append(len(stocks))

    rows = []
    for sector, values in sorted(annual_returns.items()):
        values = np.array(values)
        counts = np.array(stock_counts[sector])
        rows.append({
            'Sector': sector,
            'Mean_Annual_Return': round(values.mean() * 100, 1),
            'Median_Annual_Return': round(float(np.median(values)) * 100, 1),
            'Std_Dev': round(values.std() * 100, 1),
            'Min_Return': round(values.min() * 100, 1),
            'Max_Return': round(values.max() * 100, 1),
            'Sharpe_Ratio': round(values.mean() / values.std(), 2) if values.std() > 0 else 0,
            'Pct_Positive_Years': round((values > 0).sum() / len(values) * 100, 1),
            'Avg_Num_Stocks': round(counts.mean(), 1),
            'Num_Years': len(values),
        })

    return pd.DataFrame(rows).sort_values('Mean_Annual_Return', ascending=False)
