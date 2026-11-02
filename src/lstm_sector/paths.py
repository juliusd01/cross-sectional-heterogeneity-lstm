"""Project paths and the registry of trained model specifications.

Every analysis script imports from here rather than hard-coding a path, so the
repository can be cloned and run from any location.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- inputs -----------------------------------------------------------------
DATA = ROOT / 'data'
PRICES = DATA / 'prices'                      # test_{year}.csv, not redistributable
SEQUENCES = DATA / 'sequences'                # built by build_sequences.py
COMPANY_SECTOR = DATA / 'company_sector.csv'
MARKET_RETURN = DATA / 'market_return_french.csv'
COVARIATES = DATA / 'covariates.csv'

# --- artifacts --------------------------------------------------------------
MODELS = ROOT / 'models'                      # fitted weights, committed
METRICS = ROOT / 'metrics'                    # daily portfolio metrics, committed
PREDICTIONS = ROOT / 'predictions'            # raw per-stock predictions, not committed

# --- outputs ----------------------------------------------------------------
TABLES = ROOT / 'tables'
FIGURES = ROOT / 'figures'

TRADING_DAYS = 252
STUDY_YEARS = range(1998, 2025)               # 27 out-of-sample study periods

# Ten TRBC economic sectors, in the order `pandas.Categorical` assigns codes
# during training (alphabetical).  Academic & Educational Services never
# survives the index-constituency filter, so the embedding matrix has 10 rows.
SECTORS = [
    'Basic Materials',
    'Consumer Cyclicals',
    'Consumer Non-Cyclicals',
    'Energy',
    'Financials',
    'Healthcare',
    'Industrials',
    'Real Estate',
    'Technology',
    'Utilities',
]

SECTORS_11 = ['Academic & Educational Services'] + SECTORS


class Spec:
    """One trained model version.

    `key`    short name used on the command line and in output tables
    `path`   '<family>/<run directory>' under models/, metrics/ and predictions/
    `family` label used to group rows in the robustness table
    `layers` number of LSTM layers (None for the tree benchmarks)
    `note`   LaTeX note describing the hyper-parameters that differ
    `main`   marked with a dagger in Table 6 as the reported specification
    `in_table` False for versions that were trained but left out of Table 6
    """

    def __init__(self, key, path, family, layers, note='', main=False,
                 in_table=True):
        self.key, self.path, self.family = key, path, family
        self.layers, self.note, self.main = layers, note, main
        self.in_table = in_table

    @property
    def models(self) -> Path:
        return MODELS / self.path

    @property
    def metrics(self) -> Path:
        return METRICS / self.path

    @property
    def predictions(self) -> Path:
        return PREDICTIONS / self.path

    def __repr__(self) -> str:
        return f'Spec({self.key})'


# Order here is the order of appearance in the robustness table (Table 6).
SPECS = [
    Spec('base_l1',        'base/l1',  'LSTM Base',   1),
    Spec('base_l2',        'base/l2',  'LSTM Base',   2),
    Spec('base_l3',        'base/l3',  'LSTM Base',   3, main=True),
    Spec('base_l5',        'base/l5',  'LSTM Base',   5),
    # trained but omitted from the published Table 6 (it collapses: SR < 0)
    Spec('base_l10',       'base/l10', 'LSTM Base',  10, in_table=False),

    Spec('sector_l1_d5',   'sector/l1_D5', 'LSTM Sector', 1, r'$d_{emb}=5$'),
    Spec('sector_l2_d2',   'sector/l2_D2', 'LSTM Sector', 2, r'$d_{emb}=2$'),
    Spec('sector_l3_d2',   'sector/l3_D2', 'LSTM Sector', 3, r'$d_{emb}=2$', main=True),
    Spec('sector_l5_d5',   'sector/l5_D5', 'LSTM Sector', 5, r'$d_{emb}=5$'),

    Spec('cov_l1',         'cov/h_25_l_1_lr_0.001_B256_49383343',          'LSTM Cov', 1, r'$B=256$'),
    Spec('cov_l2',         'cov/h_25_l_2_rd_0.5_id_0.2_B128_49553001',     'LSTM Cov', 2, r'drop. 0.5/0.2'),
    Spec('cov_l2_novix',   'cov/h_25_l_2_rd_0.1_id_0.1_vix_B128_49702506', 'LSTM Cov', 2, r'drop. 0.1/0.1, VIX'),
    Spec('cov_l3',         'cov/h_25_l_3_lr_0.001_49381257',               'LSTM Cov', 3),

    Spec('covsec_l1',      'cov_sector/h_25_l_1_lr_0.001_D2',              'LSTM Cov+Sector', 1, r'$d_{emb}=2$'),
    Spec('covsec_l2',      'cov_sector/h_25_l_2__B64_D2_49553298',         'LSTM Cov+Sector', 2, r'$d_{emb}=2$'),
    Spec('covsec_l3',      'cov_sector/h_25_l_3__B64_do_0.5_D2_49553301',  'LSTM Cov+Sector', 3, r'$d_{emb}=2$, drop. 0.5'),
    Spec('covsec_l5',      'cov_sector/h_25_l_5__B64_do_0.5_D2_49553304',  'LSTM Cov+Sector', 5, r'$d_{emb}=2$, drop. 0.5'),

    Spec('rf_n100',        'rf/n100_d10_ms10',    'Random Forest', None, r'$n=100$, $d=10$'),
    Spec('rf_n500',        'rf/n500_d10_ms10',    'Random Forest', None, r'$n=500$, $d=10$', main=True),
    Spec('rf_n1000',       'rf/n1000_d20_ms10',   'Random Forest', None, r'$n=1000$, $d=20$'),
    Spec('hgb',            'hgb/iter100_lr0.1_d10', 'Grad. Boosting', None, r'$100$ iter., $d=10$'),
]

BY_KEY = {s.key: s for s in SPECS}
BY_PATH = {s.path: s for s in SPECS}

LSTM_FAMILIES = ['LSTM Base', 'LSTM Sector', 'LSTM Cov', 'LSTM Cov+Sector']

# The four headline specifications compared in Figures 6-8 and Tables 2, 3
# and 7, with the display names the paper uses.  These are also the specs for
# which metrics/ carries the k = 5, 50, 100, 200 sweep behind Figure 6.
MAIN_LABELS = {
    'sector_l3_d2': 'Sector LSTM',
    'base_l3': 'Base LSTM',
    'cov_l1': 'Cov LSTM',
    'rf_n500': 'RF',
}


def spec(key: str) -> Spec:
    """Look up a specification by its short key, with a helpful error."""
    try:
        return BY_KEY[key]
    except KeyError:
        raise SystemExit(
            f"unknown model spec {key!r}; choose one of: {', '.join(BY_KEY)}"
        ) from None
