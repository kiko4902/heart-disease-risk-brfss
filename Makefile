.PHONY: setup data notebooks test lint app docker clean

ifeq ($(OS),Windows_NT)
BIN := .venv/Scripts
else
BIN := .venv/bin
endif
PY := $(BIN)/python
# Interpreter used to create the venv when uv is not installed.
PYTHON ?= python3.12

setup:  ## Create .venv (Python 3.12) and install pinned dependencies + the cvd package
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.12 .venv && \
		uv pip install --python .venv -r requirements.txt -e . ; \
	else \
		$(PYTHON) -m venv .venv && \
		$(PY) -m pip install -r requirements.txt -e . ; \
	fi

data:  ## Download CVD_cleaned.csv into data/
	$(PY) scripts/download_data.py

notebooks:  ## Execute all notebooks top to bottom on a fresh kernel (also trains and saves the models)
	$(PY) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 \
		notebooks/01_eda.ipynb notebooks/02_statistical_analysis.ipynb \
		notebooks/03_logistic_regression.ipynb notebooks/04_predictive_model.ipynb

test:  ## Run the unit tests
	$(PY) -m pytest

lint:  ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

app:  ## Start the Streamlit app (needs models/ from `make notebooks`)
	$(PY) -m streamlit run app/app.py

docker:  ## Build the app image (needs models/ from `make notebooks`)
	docker build -t heart-disease-risk-app .

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
