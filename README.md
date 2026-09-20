# TokenFlow

An LLM efficiency gateway that sits between your app and Gemini. It reduces token usage and cost through semantic caching, smart RAG context selection, and model routing — while measuring the quality tradeoff.

## Architecture

```text
App → TokenFlow Gateway → [Cache | RAG Select | Router] → Gemini → Metrics
```

## Quick start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### 3. Start Postgres (for semantic cache)

```bash
docker compose up -d
```

### 4. Run the server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Health check

```bash
curl http://localhost:8000/health
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /chat` | Baseline Gemini passthrough |
| `POST /chat/optimized` | Full optimization pipeline |
| `GET /config` | Current feature flags |
| `GET /cache/stats` | Cache entry count |

## Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a mutex?"}'
```

## Project structure

```text
TokenFlow/
├── backend/          # FastAPI gateway
├── database/       # Postgres + pgvector
├── evaluation/     # Benchmarks, judge, dataset
├── dashboard/      # Results visualization
└── tests/          # Unit tests
```
