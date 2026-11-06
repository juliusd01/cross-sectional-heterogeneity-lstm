# Cluster job scripts

The models were trained on the Lichtenberg II cluster at TU Darmstadt. Each
script below submits one architecture at the specification reported in the main
results. The remaining rows of Table 6 re-run the same scripts with different
`--num_layers`, `--embedding_dim` and dropout settings; the values are encoded
in the run-directory names under `models/` (see the model inventory in the top
level README).

| script | architecture | training entry point |
| --- | --- | --- |
| `run_base_lstm.sbatch` | returns only | `src/train.py` |
| `run_sector_lstm.sbatch` | returns + sector embedding | `src/train.py` |
| `run_covariates_lstm.sbatch` | returns + 5 macro covariates | `src/train_dropout.py` |
| `run_covariates_sector_lstm.sbatch` | covariates + sector embedding | `src/train.py` |
| `run_tabular.sbatch` | random forest / gradient boosting | `src/train_tabular.py` |

`src/train.py` uses `nn.LSTM` with the standard inter-layer dropout.
`src/train_dropout.py` uses an explicit `LSTMCell` loop so that input
dropout and recurrent dropout can be applied separately, which the covariate
models need. Both write weights to `models/` and predictions to `predictions/`.

To run locally instead, drop the `#SBATCH` header and call the same
`python -m ...` line.
