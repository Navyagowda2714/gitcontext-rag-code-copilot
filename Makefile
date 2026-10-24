PY ?= python
INDEX ?= .cache/index
SAMPLE ?= .cache/shopfront-api

.PHONY: install test lint format demo eval index serve docker clean

install:  ## Install the package with dev extras (editable)
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the test suite (offline, no network)
	$(PY) -m pytest -q

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

demo:  ## Build the sample repo, index it and answer a few questions (offline by default)
	$(PY) examples/demo.py

eval:  ## Retrieval ablation + answer metrics on the sample repo; writes evals/results.json
	$(PY) -m gitcontext.cli eval --out evals/results.json

index:  ## Materialise the sample repo and index it into $(INDEX)
	$(PY) -m gitcontext.cli sample-repo $(SAMPLE)
	$(PY) -m gitcontext.cli index $(SAMPLE) --index-dir $(INDEX)

serve: index  ## Serve the FastAPI app on :8000 over the sample index
	$(PY) -m gitcontext.cli serve --index-dir $(INDEX)

docker:
	docker build -t gitcontext .

clean:
	rm -rf .cache .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
