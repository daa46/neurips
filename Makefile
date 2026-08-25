PYTHON ?= python3
OUTPUT_DIR ?= results/reproduction
BOOTSTRAP ?= 59
REPETITIONS ?= 2

.PHONY: help setup test lint example smoke figures reproduce

help:
	@echo "make setup      Install locked development and paper dependencies"
	@echo "make test       Run the complete unit-test suite"
	@echo "make lint       Run Ruff over source, simulations, scripts, and tests"
	@echo "make example    Run HRMS on the committed synthetic loss table"
	@echo "make smoke      Run the deterministic lightweight simulation grid"
	@echo "make figures    Render descriptive figures from the smoke grid"
	@echo "make reproduce  Verify and rebuild the complete lightweight artifact"

setup:
	uv sync --locked --extra dev --extra paper

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check applications hrms simulations scripts tests

example:
	$(PYTHON) scripts/run_hrms.py examples/synthetic_losses.csv \
		--model-registry examples/synthetic_model_registry.csv \
		--output $(OUTPUT_DIR)/example_hrms --bootstrap $(BOOTSTRAP) --seed 42

smoke:
	$(PYTHON) simulations/run_grid.py --preset smoke \
		--output-dir $(OUTPUT_DIR)/simulations --n-bootstrap $(BOOTSTRAP) \
		--repetitions $(REPETITIONS) --seed 42

figures:
	MPLCONFIGDIR=/tmp/hrms-matplotlib-cache $(PYTHON) scripts/make_figures.py \
		--simulations-dir $(OUTPUT_DIR)/simulations --output-dir $(OUTPUT_DIR)/figures

reproduce:
	PYTHON_BIN=$(PYTHON) HRMS_BOOTSTRAP=$(BOOTSTRAP) HRMS_REPETITIONS=$(REPETITIONS) \
		bash scripts/reproduce_paper.sh $(OUTPUT_DIR)
