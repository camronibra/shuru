.DEFAULT_GOAL := help
UV ?= uv
PY_DIRS := src tests scripts bench

.PHONY: help lock sync fmt lint typecheck test check demo bench schema requirements clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-13s %s\n", $$1, $$2}'

lock: ## Re-resolve uv.lock after editing pyproject.toml (then run `make requirements`)
	$(UV) lock

sync: ## Install exactly what uv.lock says into .venv
	$(UV) sync --locked

fmt: ## Auto-format and apply safe lint fixes
	$(UV) run ruff format $(PY_DIRS)
	$(UV) run ruff check --fix $(PY_DIRS)

lint: ## Check formatting and lint (no changes)
	$(UV) run ruff format --check $(PY_DIRS)
	$(UV) run ruff check $(PY_DIRS)

typecheck: ## mypy --strict
	$(UV) run mypy $(PY_DIRS)

test: ## Run the test suite
	$(UV) run pytest

check: lint typecheck test ## Everything CI checks

demo: ## Run the whole pipeline on synthetic sites served from localhost
	$(UV) run python scripts/demo.py

bench: ## Benchmark the text pipeline on this machine
	$(UV) run python bench/bench_text.py --docs 300

schema: ## Regenerate db/schema.sql from the migrations
	$(UV) run shuru schema > db/schema.sql

requirements: ## Regenerate requirements.lock (hash-pinned, for server installs) from uv.lock
	$(UV) export --no-dev --no-emit-project --format requirements-txt -o requirements.lock

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
