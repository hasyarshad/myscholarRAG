.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help setup dev fmt lint typecheck test test-all check \
	db-setup db-up db-down db-shell ollama-setup ollama-up ollama-down up down \
	ingest ingest-force sync sync-check serve \
	eval eval-fast eval-generation eval-judge eval-shipped corpus-check papers clean \
	docker-build compose-up compose-restore compose-down \
	deploy supabase-setup supabase-check push-index

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies (core + dev)
	uv sync --group dev

dev:  ## Install everything, including the local GPU and serving extras
	uv sync --group dev --extra local --extra serve

fmt:  ## Format code
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint:  ## Check formatting and lint rules
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run unit tests (no GPU, network or database)
	uv run pytest -m "not integration"

test-all:  ## Run every test, including integration
	uv run pytest

check: lint typecheck test  ## Everything CI runs

## --- Local services --------------------------------------------------------
## Postgres and Ollama run as ordinary user processes, so none of this needs root.

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

## --- Corpus ----------------------------------------------------------------

ingest:  ## Build the index from papers/ (needs make up first)
	uv run myscholar ingest run

ingest-force:  ## Rebuild the index, ignoring the content-hash cache
	uv run myscholar ingest run --force

sync:  ## Index anything new and reconcile metadata against Crossref
	uv run myscholar sync --apply

sync-check:  ## Report what sync would change, without changing anything
	uv run myscholar sync --no-ingest

serve:  ## Run the API on :8000
	uv run myscholar serve

## --- Evaluation ------------------------------------------------------------

eval:  ## Run every evaluation and write results/report.md (~30 min)
	uv run python scripts/eval.py

eval-fast:  ## Everything except generation; deterministic, about two minutes
	uv run python scripts/eval.py --skip generation

eval-generation:  ## The generation ablation, every configuration
	uv run python scripts/eval_generation.py

eval-judge:  ## Check the faithfulness judge against known cases
	uv run python scripts/eval_judge.py

eval-shipped:  ## The shipped pipeline only, no baselines
	uv run python scripts/eval_generation.py --configs adaptive

corpus-check:  ## Assert the invariants a healthy corpus satisfies
	uv run myscholar papers check

papers:  ## List what is indexed
	uv run myscholar papers list
	@echo
	uv run myscholar papers sections

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

## --- Containers ------------------------------------------------------------
## The compose stack runs its own Postgres on 5434, separate from the
## development database on 5433.

docker-build:  ## Build the production image (exports and verifies the embedder)
	docker build -f docker/Dockerfile -t myscholar-api:local .

compose-up:  ## Bring up the containerised stack (db + api) on :8001
	docker compose -f docker/compose.yaml up --build -d
	@echo "waiting for the API..."
	@until curl -sf http://localhost:8001/healthz >/dev/null; do sleep 2; done
	@curl -s http://localhost:8001/readyz; echo

compose-restore:  ## Load the local index into the compose database
	./scripts/push_index.sh --dump-only
	docker compose -f docker/compose.yaml exec -T db \
		psql -U myscholar -d myscholar -v ON_ERROR_STOP=1 < artifacts/index.sql
	@curl -s http://localhost:8001/v1/corpus; echo

compose-down:  ## Stop the stack and remove its database volume
	docker compose -f docker/compose.yaml down -v

## --- Deployment ------------------------------------------------------------

deploy:  ## Build on Cloud Build and deploy to Cloud Run (reads .env)
	./scripts/deploy_cloudrun.sh

supabase-setup:  ## Create the schema and search indexes on Supabase
	./scripts/supabase_setup.sh

supabase-check:  ## Report what is on Supabase, changing nothing
	./scripts/supabase_setup.sh --check

push-index:  ## Load the local index into Supabase (reads SUPABASE_DB_URL)
	./scripts/push_index.sh
