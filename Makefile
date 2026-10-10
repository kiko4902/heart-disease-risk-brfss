.PHONY: setup data notebooks test lint app docker lock clean

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
		--ExecutePreprocessor.record_timing=False \
		notebooks/01_eda.ipynb notebooks/02_statistical_analysis.ipynb \
		notebooks/03_logistic_regression.ipynb notebooks/04_predictive_model.ipynb

test:  ## Run the unit tests
	$(PY) -m pytest

lint:  ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

app:  ## Start the Streamlit app (uses the versioned models in app/model/)
	$(PY) -m streamlit run app/app.py

docker:  ## Build the slim app image
	docker build -t heart-disease-risk-app .

lock:  ## Recompile both lock files (needs uv); the app lock is constrained to the main lock
	uv pip compile requirements.in --universal --python-version 3.12 -o requirements.txt
	uv pip compile requirements-app.in -c requirements.txt --universal --python-version 3.12 -o requirements-app.txt
	cp requirements-app.txt app/requirements.txt

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
