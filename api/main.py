"""FastAPI application entrypoint."""
from fastapi import FastAPI

app = FastAPI(title="AI Pipeline Debugger")


@app.get("/health")
def health():
    return {"status": "ok"}
