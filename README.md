# Cross-Sectional Heterogeneity in LSTM Networks for Financial Time Series

Replication code for the corresponding paper.

Fitted weights for all 21 model specifications and the daily portfolio metrics
are included, so no retraining is needed to regenerate the tables and figures.

## Quickstart

```bash
pip install -r requirements.txt
make from-metrics     # everything that needs no licensed data
```

Scripts also run standalone with `PYTHONPATH=src`, e.g.
`PYTHONPATH=src python analysis/tab07_performance.py`.

## Required Data

To reproduce all the results, one needs
- The index of leavers and joiners for the S&P 500, which can be retrieved from LSEG (formerly Thompson Reuters)
- Daily prices of the constitutents of the S&P 500. A list with tickers that need to be downloaded from LSEG is [here](data/tickers_lseg.txt)
- the observations from the 5 covariates presented in the paper from January 1st, 1995 onwards.
- running the following code gives the input sequences for the LSTM models:

```sh
python data/add_features.py --oil data/covariates/oil.csv --gold data/covariates/gold.csv --vix data/covariates/vix.csv --rates data/covariates/rates.csv --sectors data/company_sector.csv
```

## Reproducing each table and figure

All outputs are committed in `tables/` and `figures/`. `make help` lists
individual targets.

| Output | Command |
| --- | --- |
| Table 2 — Diebold-Mariano | `make tab02` |
| Table 3 — Ledoit-Wolf Sharpe tests | `make tab03` |
| Table 6 — robustness | `make tab06` |
| Table 7 — risk and return characteristics | `make tab07` |
| Figure 6 — performance by portfolio size | `make fig06` |
| Figures 7, 8 — return and risk radars | `make fig07 fig08` |
| Figure 9 — metrics by year | `make fig09` |
| Figure 12 — sector embeddings | `make fig12` |

These additionally need the Refinitiv price panels in `data/prices/`:

| Output | Command |
| --- | --- |
| Table 1 — sector summary statistics | `make tab01` |
| Tables 4, 5 — sector contributions | `make tab04_05` |
| Figures 10, 11 — embedding quality | `make fig10_11` |
| Figures 13, 14 — sector selection | `make fig13_14` |
| Figure 15 — short-term reversal | `make fig15` |

Figures 1–4 are manuscript diagrams. Figure 5 needs the Bloomberg/Cboe/LSEG
covariate series and is not included.

## License

MIT (`LICENSE`). The data in `data/README.md` is not covered by it.
