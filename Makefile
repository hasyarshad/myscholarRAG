.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help setup dev fmt lint typecheck test test-all db-up db-down clean

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

## --- Local services -------------------------------------------------------
## Postgres and Ollama both run as ordinary user processes. The workstation is
## a shared lab machine with no root, so Docker is not available here; the
## Compose file under docker/ is a packaging artifact for other people.

db-setup:  ## Create the Postgres cluster and enable pgvector (run once)
	./scripts/db.sh setup

db-up:  ## Start Postgres
	./scripts/db.sh start

db-down:  ## Stop Postgres
	./scripts/db.sh stop

db-shell:  ## Open a psql session
	./scripts/db.sh psql

ollama-setup:  ## Install Ollama into ~/.local and pull the models (run once)
	./scripts/ollama.sh install
	./scripts/ollama.sh models

ollama-up:  ## Start the Ollama server
	./scripts/ollama.sh start

ollama-down:  ## Stop the Ollama server
	./scripts/ollama.sh stop

up: db-up ollama-up  ## Start every local service
	@echo "services up"

down: db-down ollama-down  ## Stop every local service
.PHONY: db-setup db-shell ollama-setup ollama-up ollama-down up down ingest ingest-force papers serve

## --- Corpus ---------------------------------------------------------------

ingest:  ## Build the index from papers/ (needs make up first)
	uv run myscholar ingest run

ingest-force:  ## Rebuild the index, ignoring the content-hash cache
	uv run myscholar ingest run --force

sync:  ## Index anything new and reconcile metadata against the publisher record
	uv run myscholar sync --apply

sync-check:  ## Same, but report only — changes nothing
	uv run myscholar sync --no-ingest

serve:  ## Run the API on :8000
	uv run myscholar serve

papers:  ## List what is indexed
	uv run myscholar papers list
	@echo
	uv run myscholar papers sections

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
