# AI Pipeline Debugger

LLM-assisted triage for failed data-pipeline runs (Spark, Airflow, dbt). Paste a log,
get a diagnosis, a suggested fix, and a list of historically similar incidents — backed
by a local vector store and a confidence-aware router that prefers a local Ollama model
and falls back to OpenAI only when needed.

## Architecture

```
            ┌──────────────┐    ┌────────────────┐
   logs ───▶│  Ingestion   │───▶│  Parser (RX +  │
            │  collectors  │    │  LLM fallback) │
            └──────────────┘    └───────┬────────┘
                                        ▼
                               ┌──────────────────┐
                               │ Vector DB index  │  (ChromaDB, cosine)
                               └────────┬─────────┘
                                        ▼
   request ─▶ FastAPI /analyze ─▶ Retriever ─▶ Prompt builder ─▶ LLM Router
                                                                  │
                                              ┌───────────────────┼──────────────────┐
                                              ▼                                      ▼
                                       Ollama (local)                             OpenAI
                                       confidence score ──low──────────────────▶ fallback
```

Layered per Clean Architecture: `api/` (transport), `core/` (config/logging),
`ingestion/` & `vector_db/` (infrastructure), `llm/` (domain ports + adapters).

## LLM routing modes

Set `LLM_MODE` in `.env`:

| Mode | Behaviour |
|------|-----------|
| `cheap` | Ollama only — no fallback, zero API cost |
| `accurate` | OpenAI only |
| `auto` *(default)* | Ollama first; falls back to OpenAI when `confidence < LLM_CONFIDENCE_THRESHOLD` or on error |

---

## Quickstart — local

### 1. Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/download) installed and running

```bash
ollama serve                  # start Ollama daemon
ollama pull llama3.1:8b       # download model (~4.7 GB, one-time)
```

### 2. Install

```bash
git clone https://github.com/<your-username>/ai-pipeline-debugger.git
cd ai-pipeline-debugger

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -e ".[dev,ui]"
```

### 3. Configure

```bash
cp .env.example .env
```

Minimum required in `.env`:

```ini
LLM_MODE=auto
OLLAMA_MODEL=llama3.1:8b
OPENAI_API_KEY=sk-...         # only needed for accurate/auto fallback
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

API is live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

---

## Quickstart — Docker

```bash
cp .env.example .env          # add OPENAI_API_KEY if needed
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Ollama | http://localhost:11434 |

---

## Usage examples

### Analyze a Spark OOM failure

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "etl_orders_daily",
    "stage": "join_users_orders",
    "log_excerpt": "[2025-04-01 03:14:21] ERROR Executor: Exception in task 0.0 in stage 4.0\norg.apache.spark.SparkException: Job aborted due to stage failure\njava.lang.OutOfMemoryError: Java heap space\n  at java.util.Arrays.copyOf(Arrays.java:3236)\n  at org.apache.spark.sql.execution.joins.SortMergeJoinExec.doExecute(SortMergeJoinExec.scala:382)"
  }'
```

Response:

```json
{
  "incident_id": "bdcec86f3c66493ab4b9d2aa6f7d9515",
  "error_type": "OutOfMemoryError",
  "root_cause": "The Spark executor ran out of JVM heap memory during a SortMergeJoin. Likely caused by data skew or undersized executor memory.",
  "suggested_fix": "1. Increase executor memory: spark.executor.memory=8g\n2. Enable AQE: spark.sql.adaptive.enabled=true\n3. Check join key distribution for skew",
  "severity": "high",
  "confidence": 0.9,
  "tags": ["Spark", "OutOfMemoryError", "join"],
  "similar_incidents": [],
  "llm": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "confidence": 0.9,
    "latency_ms": 21349,
    "prompt_tokens": 524,
    "completion_tokens": 197,
    "fallback_used": false
  }
}
```

### Submit feedback (closes the learning loop)

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "bdcec86f3c66493ab4b9d2aa6f7d9515",
    "helpful": true,
    "actual_fix": "Set spark.executor.memory=12g and enabled AQE. Issue resolved.",
    "notes": "Data was skewed on region_id column — 30% of rows had NULL."
  }'
```

### Check provider health

```bash
curl http://localhost:8000/health
# {"status":"ok","providers":{"ollama":true,"openai":false}}
```

---

## Streamlit UI

```bash
streamlit run ui/app.py
# → http://localhost:8501
```

Paste a log, click **Analyze**, and inspect error type, root cause, suggested fix, LLM trace and similar incidents side by side.

---

## Generate mock logs (for testing)

```bash
python scripts/generate_mock_logs.py --count 20 --out data/raw_logs
```

---

## Run tests

```bash
pytest tests/ -v
```

```
tests/test_parser.py::test_parses_python_traceback     PASSED
tests/test_parser.py::test_parses_spark_oom            PASSED
tests/test_parser.py::test_parses_airflow_exception    PASSED
tests/test_router.py::test_auto_falls_back_on_low_confidence  PASSED
tests/test_router.py::test_accurate_mode_skips_ollama  PASSED
tests/test_retriever.py::test_retrieves_semantically_similar  PASSED
...
```

## Run evaluation benchmark

```bash
# API must be running
python -m evals.benchmark
```

```
spark-oom-001   | expected=OutOfMemoryError  got=OutOfMemoryError  OK  (ollama, 2341ms)
airflow-tz-002  | expected=AirflowException  got=AirflowException  OK  (ollama, 1876ms)
py-keyerror-003 | expected=KeyError          got=KeyError          OK  (ollama, 1654ms)
```

---

## Use cases

1. **On-call triage** — paste a failing task log, receive root cause + fix in seconds.
2. **Pattern discovery** — semantic retrieval surfaces "we saw this last quarter".
3. **Knowledge capture** — `/feedback` writes verified fixes back into the vector store so future retrievals get battle-tested answers.
