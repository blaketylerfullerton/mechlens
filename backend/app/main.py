import os
from contextlib import asynccontextmanager
from dataclasses import asdict

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.engine import Engine

load_dotenv()

# Local dev defaults (CPU, float32, a small real model) — override via .env
# for a GPU box (e.g. DEVICE_MAP=auto, and drop dtype to bfloat16 in Engine).
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cpu")

engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Loaded once at startup, not per-request — reloading the model on every
    # call would make each request take as long as `inspect_trace.py`'s
    # entire startup.
    engine = Engine(MODEL_NAME, device_map=DEVICE_MAP, dtype=torch.float32)
    yield
    engine = None


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str


class GenerateRequest(PromptRequest):
    max_new_tokens: int = 30


class TraceRequest(PromptRequest):
    top_k: int = 5


class AblateRequest(PromptRequest):
    layer: int
    head: int
    max_new_tokens: int = 30


class JacobianRequest(PromptRequest):
    target_token: str | None = None
    top_k: int = 5


@app.post("/api/generate")
def generate(req: GenerateRequest):
    return {"text": engine.generate(req.prompt, max_new_tokens=req.max_new_tokens)}


@app.post("/api/trace")
def trace(req: TraceRequest):
    return asdict(engine.trace(req.prompt, top_k=req.top_k))


@app.post("/api/ablate")
def ablate(req: AblateRequest):
    return {"text": engine.ablate_head(req.prompt, req.layer, req.head, max_new_tokens=req.max_new_tokens)}


@app.post("/api/jacobian")
def jacobian(req: JacobianRequest):
    return asdict(engine.jacobian_lens(req.prompt, target_token=req.target_token, top_k=req.top_k))
