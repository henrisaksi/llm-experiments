# llm-experiments

A containerized environment for experimenting with, serving, and chatting with local Large Language Models—specifically designed around the **Gemma 4** architecture.

---

## Features

- **Gemma 4 Implementation (`src/gemma4`)**: Standalone Hugging Face-compatible implementation supporting configurations, tokenization, and conditional generation.
- **LLM Service (`services/llm_service`)**: FastAPI backend providing buffered (`/generate`) and streaming (`/generate/stream`) inference, health checks, and dynamic model swapping in memory.
- **Interactive UI (`services/ui_service`)**: Chainlit web interface with real-time token streaming and on-the-fly model switching via chat settings.
- **Multi-Device Support**: Automatic hardware fallback order: CUDA $\rightarrow$ Apple Silicon (MPS) $\rightarrow$ CPU.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   UI Service (Chainlit)     |
                      |   http://localhost:8080     |
                      +--------------+--------------+
                                     | HTTP / Stream
                                     v
                      +-----------------------------+
                      |   LLM Service (FastAPI)     |
                      |   http://localhost:8000     |
                      +--------------+--------------+
                                     | Loads via Gemma4ForConditionalGeneration
                                     v
                      +-----------------------------+
                      |       ./models/<model>      |
                      +-----------------------------+
```

---

## Quick Start (Docker Compose)

Start both the LLM engine and the UI frontend with Docker Compose:

```bash
docker compose up --build
```

- **Web UI**: [http://localhost:8080](http://localhost:8080)
- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Downloading Models

The service loads weights dynamically from the `./models/` directory. All models must be compatible with the **Gemma 4** architecture.

To download checkpoints directly from Hugging Face:

```bash
uv run python scripts/download_model.py <hf_repo_id> [--name <folder_name>]
```

**Examples:**
```bash
# Download official Gemma 4 instruct checkpoint
uv run python scripts/download_model.py google/gemma-4-E2B-it

# Download fine-tuned / uncensored variant
uv run python scripts/download_model.py TrevorJS/gemma-4-E2B-it-uncensored --name gemma-4-E2B-it-uncensored
```

Any subdirectory in `./models/` containing `.safetensors` or `.bin` weight files will automatically appear in the UI model selection dropdown and the `/models` API endpoint.

---

## Local Development (Without Docker)

### Prerequisites
- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/) package manager

### Setup
```bash
# Install dependencies
uv sync

# Run LLM service
MODELS_DIR=./models uv run uvicorn services.llm_service.server:app --host 0.0.0.0 --port 8000 --reload

# Run Chainlit UI (in another terminal)
LLM_SERVICE_URL=http://localhost:8000 uv run chainlit run services/ui_service/app.py --port 8080
```
