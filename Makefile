.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help setup dev fmt lint typecheck test test-all db-up db-down db-logs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies (core + dev)
	uv sync --group dev

dev:  ## Install everything, including local GPU and serving extras
	uv sync --group dev --extra local --extra serve

fmt:  ## Format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:  ## Check formatting and lint rules
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run unit tests (no GPU, no network, no database)
	uv run pytest -m "not integration"

test-all:  ## Run every test, including integration
	uv run pytest

check: lint typecheck test  ## Everything CI runs
.PHONY: check

db-up:  ## Start Postgres with pgvector
	docker compose -f docker/compose.yaml up -d
	@echo "waiting for postgres..."
	@until docker compose -f docker/compose.yaml exec -T db pg_isready -U myscholar -q; do sleep 1; done
	@echo "ready on localhost:5433"

db-down:  ## Stop Postgres
	docker compose -f docker/compose.yaml down

db-logs:  ## Tail Postgres logs
	docker compose -f docker/compose.yaml logs -f db

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
