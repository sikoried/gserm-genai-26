# Oracle — common dev tasks. Run `make` or `make help` for the list.

VENV    := .venv
PY      := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
BACKEND := $(UVICORN) oracle.api:app --host 127.0.0.1 --port 8000 --reload --reload-dir oracle --reload-dir configs

.DEFAULT_GOAL := help
.PHONY: help setup dev backend frontend test clean

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Create the venv and install backend + frontend deps
	python3.12 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	npm --prefix frontend install

dev: ## Run backend (:8000) + frontend (:5173) together; Ctrl-C stops both
	@echo "backend  -> http://localhost:8000   (FastAPI, reloads on oracle/ + configs/)"
	@echo "frontend -> http://localhost:5173   (Vite, proxies /api -> :8000)"
	@trap 'kill 0' INT TERM EXIT; \
		$(BACKEND) & \
		npm --prefix frontend run dev & \
		wait

backend: ## Run only the FastAPI backend (:8000)
	$(BACKEND)

frontend: ## Run only the Vite dev server (:5173)
	npm --prefix frontend run dev

test: ## Run the Python test suite
	$(PY) -m pytest -q tests/

clean: ## Remove Python caches
	rm -rf .pytest_cache
	find oracle bin tests -type d -name __pycache__ -exec rm -rf {} +
