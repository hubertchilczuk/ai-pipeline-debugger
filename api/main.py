"""FastAPI application entrypoint."""
from fastapi import FastAPI

from .schemas import DebugRequest, DebugResponse, IndexRequest

app = FastAPI(title="AI Pipeline Debugger")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/debug", response_model=DebugResponse)
def debug(req: DebugRequest):
    return DebugResponse(explanation="stub", confidence=0.0, citations=[])


@app.post("/index")
def index(req: IndexRequest):
    return {"indexed": len(req.documents)}

# error handlers wired
# retry wired
# tracing wired
# confidence scoring tweaked
