PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

# `make figures-from-metrics` and
# `make tables-from-metrics` run on a clean clone; the `*-from-prices` targets
# additionally need the Refinitiv panels in data/prices/ (see data/README.md).

.PHONY: help all from-metrics from-prices figures-from-metrics tables-from-metrics \
        figures-from-prices tables-from-prices sequences clean-outputs

help:
	@echo "make from-metrics   reproduce everything that needs only the committed artifacts"
	@echo "make from-prices    reproduce the rest (needs data/prices/)"
	@echo "make all            both of the above, in dependency order"
	@echo "make covariates     assemble data/covariates.csv from the vendor extracts"
	@echo "make sequences      rebuild the model inputs from the raw price panels"
	@echo ""
	@echo "Individual outputs:  make tab02  fig07  fig13_14  ... (see below)"
	@$(MAKE) --no-print-directory list

list:
	@grep -E '^(tab|fig)[0-9_]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

all: from-metrics from-prices

from-metrics: tables-from-metrics figures-from-metrics
from-prices:  tables-from-prices  figures-from-prices

# --- needs only metrics/, models/ and data/market_return_french.csv ---------
tables-from-metrics: tab02 tab03 tab06 tab07
figures-from-metrics: fig06 fig07 fig08 fig09 fig12

# --- additionally needs the raw price panels --------------------------------
tables-from-prices: tab01 tab04_05
figures-from-prices: fig10_11 fig13_14 fig15

# Table 7 feeds Figures 7 and 8; Figure 9 feeds the embedding-quality analysis.
fig07 fig08: tab07
fig10_11: tab04_05
tab04_05: fig09

tab01:
	$(PYTHON) analysis/tab01_sector_summary.py
tab02:
	$(PYTHON) analysis/tab02_diebold_mariano.py
tab03:
	$(PYTHON) analysis/tab03_ledoit_wolf.py
tab04_05:
	$(PYTHON) analysis/tab04_05_sector_contribution.py --year 2007 --year 2003
	$(PYTHON) analysis/tab04_05_sector_contribution.py --all-years
tab06:
	$(PYTHON) analysis/tab06_robustness.py
tab07:
	$(PYTHON) analysis/tab07_performance.py

fig06:
	$(PYTHON) analysis/fig06_portfolio_sizes.py
# Figures 7 and 8 both lever each strategy to the market's annualised
# volatility, so return and risk are compared at one risk budget.  The scripts
# still take --scale percent / --k <n> if you want the alternatives.
fig07:
	$(PYTHON) analysis/fig07_return_radar.py --out figures/fig07_return_radar.png
fig08:
	$(PYTHON) analysis/fig08_risk_radar.py --out figures/fig08_risk_radar.png
fig09:
	$(PYTHON) analysis/fig09_annual_metrics.py
fig10_11:
	$(PYTHON) analysis/fig10_11_embedding_quality.py
fig12:
	$(PYTHON) analysis/fig12_sector_embedding.py
fig13_14:
	$(PYTHON) analysis/fig13_14_sector_selection.py
fig15:
	$(PYTHON) analysis/fig15_short_term_reversal.py

# --- model inputs -----------------------------------------------------------
SEQUENCE_VARIANTS = lag_60 lag_60_sector lag_60_cov lag_60_cov_sector tabular

covariates:
	@test -n "$(SOURCE_DIR)" || { echo "usage: make covariates SOURCE_DIR=/path/to/vendor/extracts"; exit 1; }
	$(PYTHON) -m build_covariates --source-dir $(SOURCE_DIR)

sequences:
	@for v in $(SEQUENCE_VARIANTS); do \
	    echo "building $$v ..."; \
	    $(PYTHON) -m build_sequences --variant $$v || exit 1; \
	done

# Only removes what `make from-metrics` can put back, so running it on a machine
# without the licensed data cannot destroy a committed figure.  Use
# clean-all-outputs when you do have the data and want a full rebuild.
METRICS_FIGURES = fig06_daily_performance_by_k fig07_return_radar fig08_risk_radar \
                  fig09_annual_metrics fig12_sector_embedding_2007
METRICS_TABLES  = tab02_diebold_mariano tab03_ledoit_wolf tab03_ledoit_wolf_individual \
                  tab03_ledoit_wolf_pairwise tab06_robustness tab06_robustness_lstm \
                  tab06_robustness_benchmark tab07_return_characteristics \
                  tab07_risk_characteristics annual_metrics_summary \
                  portfolio_size_metrics_daily robustness_annual_metrics

clean-outputs:
	@for f in $(METRICS_FIGURES); do rm -f figures/$$f.png; done
	@for f in $(METRICS_TABLES); do rm -f tables/$$f.csv tables/$$f.xlsx tables/$$f.tex; done
	@echo "removed the outputs that 'make from-metrics' regenerates"

clean-all-outputs:
	rm -f figures/*.png tables/*.csv tables/*.xlsx tables/*.tex
	@echo "removed every generated output; 'make all' needs the licensed data to restore them"
