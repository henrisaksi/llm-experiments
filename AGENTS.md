# Agent Guidelines for llm-experiments

This document defines guidelines, architectural overview, conventions, and operational workflows for AI coding agents working within the `llm-experiments` repository.

---

## 1. Project Overview

`llm-experiments` is a containerized and modular Python project designed for running, experimenting with, and serving local Large Language Models (specifically Gemma 4 architectures and variations) with interactive UI interfaces.

### Core Components
- **`src/gemma4/`**: Custom implementation modules for Gemma 4 (configurations, modeling, processing, tokenization, feature extraction).
- **`services/llm_service/`**: FastAPI backend service exposing REST endpoints for model inference, streaming generation, model switching, and health checks.
- **`services/ui_service/`**: Chainlit-based conversational UI frontend that connects to `llm_service` and provides user controls for model selection and parameters.
- **`models/`**: Local directory housing model weights and checkpoints (e.g., `gemma-4-E2B-it`, `gemma-4-E2B-it-uncensored`).
- **`scripts/`**: Automation scripts (e.g., `download_model.py` for pulling checkpoints from Hugging Face).
- **`docker-compose.yml`**: Multi-container orchestration linking `llm_service` (port 8000) and `ui_service` (port 8080).

---

## 2. Environment & Package Management

- **Python Version**: `>= 3.12` (managed via `.python-version` and `uv`).
- **Package Manager**: `uv` (uses `pyproject.toml` and `uv.lock`).
- **Primary Dependencies**:
  - `torch`, `torchvision`, `accelerate`
  - `transformers`, `datasets`
  - `fastapi`, `uvicorn`, `httpx`, `pydantic`
  - `chainlit`
  - `jupyterlab`, `ipywidgets`

### Running Commands
- When executing Python commands or scripts in the local environment, always run inside the managed virtual environment or use `uv run`:
  ```bash
  uv run python scripts/download_model.py <repo-id>
  ```
- Dependency updates:
  ```bash
  uv add <package-name>
  uv sync
  ```

---

## 3. Architecture & Service Conventions

### LLM Service (`services/llm_service/server.py`)
- **FastAPI** service handling model loading and inference.
- Dynamically scans `MODELS_DIR` (defaults to `/app/models` or local `./models`).
- Supports dynamic model swapping (`POST /models/load`) protected by threading locks to prevent race conditions during weight memory deallocation (`torch.cuda.empty_cache()` / MPS / GC).
- Supports both buffered (`POST /generate`) and streaming responses (`POST /generate/stream`) using `transformers.TextIteratorStreamer`.

### UI Service (`services/ui_service/app.py`)
- Built on **Chainlit**.
- Communicates with `llm_service` via HTTP client requests using `LLM_SERVICE_URL`.
- Provides model selection via `ChatSettings` and handles streaming generation chunks to user chat sessions.

### Gemma 4 Implementation (`src/gemma4/`)
- Contains standalone implementation files adhering to Hugging Face `transformers` patterns.
- When referencing or importing within services, ensure `src` is properly resolvable on `sys.path`.

---

## 4. Development & Operational Workflows

### Starting Services Locally with Docker Compose
```bash
docker compose up --build
```
- LLM Service endpoint: `http://localhost:8000` (Healthcheck: `http://localhost:8000/health`)
- UI Service interface: `http://localhost:8080`

### Adding or Downloading New Models
Use the provided download helper:
```bash
uv run python scripts/download_model.py <hf_repo_id> [--name <folder_name>]
```
Models should be placed under the `./models/` directory. Any subdirectory with valid `.safetensors` or `.bin` weight files will be automatically detected by `llm_service`.

---

## 5. Agent Instructions & Code Guidelines

1. **Adhere to Python 3.12+ idioms**: Use modern type hinting (`list[str]`, `dict[str, Any]`, `X | None`).
2. **Resource Management**: When modifying inference or model lifecycle code, ensure memory cleanup (forcing garbage collection and clearing device caches) is preserved to prevent OOM errors.
3. **Hardware Agnostic**: Maintain hardware detection fallback order: `cuda` -> `mps` -> `cpu`.
4. **Volume Mounts & Paths**: Keep path resolutions relative to project root or environment variables (`MODELS_DIR`, `LLM_SERVICE_URL`) so that both bare-metal local execution and Docker container mounts function seamlessly.
5. **Clean Commits & Edits**: Do not commit large binary files or `.safetensors` model weights to Git (verify `.gitignore`).
