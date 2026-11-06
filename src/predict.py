"""Run the saved LSTM weights on the test year of each study period, without retraining.

    python src/predict.py --model sector --spec l3_D2
    python src/predict.py --model cov --spec h_25_l_1_lr_0.001_B256_49383343 --years 2007 2008

--model picks the family (models/<model>/) and the sequence files it reads:
    base        data/sequences/lag_60/
    sector      data/sequences/lag_60_sector/
    cov         data/sequences/lag_60_cov/
    cov_sector  data/sequences/lag_60_cov_sector/
--spec is the run directory under models/<model>/ (omit it to list the choices).

The architecture (layers, hidden size, sector embedding, covariate inputs) is read
from the checkpoint itself.  Data preparation mirrors train.py exactly: the test set
is every row of returns_{Y}.parquet dated in Y, and sector ids are the category
codes of the year file's 'sector' column, as during training.

Output: predictions/<model>/<spec>/predictions_{Y}.parquet, in the same format
train.py writes, so metrics.py and the analysis scripts read it unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import models
import models_dropout
from paths import MODELS, PREDICTIONS, SEQUENCES, STUDY_YEARS

VARIANT = {'base': 'lag_60', 'sector': 'lag_60_sector',
           'cov': 'lag_60_cov', 'cov_sector': 'lag_60_cov_sector'}
COVARIATES = ['oil_return', 'vix', 't10y3m', 't10y_change', 'gold_return']


def build_model(state: dict):
    """Recreate the network that produced `state`, reading every size from the weights.
    Returns (model, input_size, num_sectors or None)."""
    hidden = state['fc.weight'].shape[1]
    kwargs = {'num_classes': 2}
    if 'sector_embedding.weight' in state:
        kwargs['num_sectors'], kwargs['sector_embedding_dim'] = state['sector_embedding.weight'].shape
        hidden -= kwargs['sector_embedding_dim']
        arch = 'heterogeneity'
    else:
        arch = 'base'
    kwargs['hidden_size'] = hidden

    if 'lstm.weight_ih_l0' in state:            # nn.LSTM (train.py)
        kwargs['input_size'] = state['lstm.weight_ih_l0'].shape[1]
        kwargs['num_layers'] = sum(k.startswith('lstm.weight_ih_l') for k in state)
        model = models.get_model(arch, dropout=0.0, **kwargs)
    else:                                       # LSTMCell stack (train_dropout.py)
        kwargs['input_size'] = state['lstm_cells.0.weight_ih'].shape[1]
        kwargs['num_layers'] = sum(k.endswith('.weight_ih') for k in state)
        model = models_dropout.get_model(arch, input_dropout=0.0, recurrent_dropout=0.0, **kwargs)
    model.load_state_dict(state)                # strict: fails loudly on any mismatch
    return model.eval(), kwargs['input_size'], kwargs.get('num_sectors')


def load_test_year(path: Path, year: int, spec: str, input_size: int, num_sectors: int | None):
    """Test-year inputs, prepared exactly as train.load_data does."""
    df = pd.read_parquet(path).sort_values('date').reset_index(drop=True)

    if input_size == 1:
        features = ['sequence']
    else:
        features = [c for c in df.columns if c.endswith('sequence')]
        # train_dropout.py --var_left_out puts the dropped covariate in the run name, e.g. '_vix_'
        left_out = [c for c in COVARIATES if f'_{c}_' in f'_{spec}_']
        features = [c for c in features if c[:-len('_sequence')] not in left_out]
    if len(features) != input_size:
        raise SystemExit(f'{path}: model expects {input_size} input series, file gives {len(features)} {features}')

    test = (df['date'].dt.year == year).to_numpy()
    X = np.stack([np.stack(df.loc[test, c].to_numpy()) for c in features], axis=-1).astype('float32')
    y = np.argmax(df.loc[test, ['Class0', 'Class1']].to_numpy(), axis=1)

    sectors = None
    if num_sectors is not None:
        codes = df['sector'].astype('category').cat.codes.to_numpy()
        if (codes < 0).any() or codes.max() + 1 != num_sectors:
            raise SystemExit(f'{path}: {df["sector"].nunique()} sectors (missing: {(codes < 0).sum()} rows), '
                             f'but the checkpoint embeds {num_sectors}')
        sectors = codes[test]

    meta = df.loc[test, ['stock', 'date', 'return']].reset_index(drop=True)
    return X, y, sectors, meta


@torch.no_grad()
def predict(model, X, sectors, device, batch_size=4096):
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out = model(xb) if sectors is None else \
            model(xb, torch.from_numpy(sectors[i:i + batch_size]).long().to(device))
        probs.append(torch.softmax(out, dim=1).cpu())
    return torch.cat(probs).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=VARIANT)
    p.add_argument('--spec', help='run directory under models/<model>/')
    p.add_argument('--years', nargs='*', type=int, default=list(STUDY_YEARS))
    p.add_argument('--data_dir', default=str(SEQUENCES), help='default: data/sequences')
    p.add_argument('--out_dir', default=str(PREDICTIONS), help='default: predictions')
    args = p.parse_args()

    specs = sorted(d.name for d in (MODELS / args.model).iterdir() if d.is_dir())
    if args.spec not in specs:
        p.error(f'--spec must be one of: {", ".join(specs)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir) / args.model / args.spec
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        state = torch.load(MODELS / args.model / args.spec / f'best_lstm_model_{year}.pth',
                           map_location='cpu', weights_only=True)
        model, input_size, num_sectors = build_model(state)
        path = Path(args.data_dir) / VARIANT[args.model] / f'returns_{year}.parquet'
        X, y, sectors, meta = load_test_year(path, year, args.spec, input_size, num_sectors)
        prob = predict(model.to(device), X, sectors, device)

        meta['pred_Class0'] = prob[:, 0]
        meta['pred_Class1'] = prob[:, 1]
        meta['predicted_class'] = prob.argmax(axis=1)
        meta['actual_class'] = y
        out = out_dir / f'predictions_{year}.parquet'
        meta.to_parquet(out, index=False)
        print(f'{year}: {len(meta):,} test rows, accuracy {(meta.predicted_class == y).mean():.4f} -> {out}')


if __name__ == '__main__':
    main()
