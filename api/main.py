"""FastAPI application entrypoint."""
from fastapi import FastAPI

from .schemas import DebugRequest, DebugResponse

app = FastAPI(title="AI Pipeline Debugger")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/debug", response_model=DebugResponse)
def debug(req: DebugRequest):
    # TODO: wire to ingestion + retrieval + llm
    return DebugResponse(explanation="stub", confidence=0.0, citations=[])
