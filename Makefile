.DEFAULT_GOAL := help
UV ?= uv
PY_DIRS := $(wildcard src tests scripts)

.PHONY: help lock sync fmt lint typecheck test check clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

lock: ## Re-resolve uv.lock after editing pyproject.toml
	$(UV) lock

sync: ## Install exactly what uv.lock says into the venv
	$(UV) sync --locked

fmt: ## Auto-format and apply safe lint fixes
	$(UV) run ruff format $(PY_DIRS)
	$(UV) run ruff check --fix $(PY_DIRS)

lint: ## Check formatting and lint (no changes)
	$(UV) run ruff format --check $(PY_DIRS)
	$(UV) run ruff check $(PY_DIRS)

typecheck: ## mypy --strict over src and tests
	$(UV) run mypy $(wildcard src tests)

test: ## Run the test suite
	$(UV) run pytest

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist