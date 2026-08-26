import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.engine import Engine
from app.lenses.ablate import ablate_head
from app.lenses.generate import build_prompt as run_build_prompt
from app.lenses.generate import chat as run_chat
from app.lenses.generate import generate as run_generate
from app.lenses.jacobian import jacobian_lens
from app.lenses.trace import trace as run_trace

load_dotenv()

# Local dev defaults (CPU, float32, a small real model) — override via .env
# for a GPU box (e.g. DEVICE_MAP=auto, and drop dtype to bfloat16 in Engine).
# The -Instruct variant (not the bare base model) is the default here
# specifically so /api/chat has a real chat template and end-of-turn token
# to work with — the base model has neither, so it can't hold a
# conversation without degrading into repetition.
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cpu")

# How many distinct models to keep loaded at once. The UI's model picker lets
# a user flip between models, but each one is real memory — evict the
# least-recently-used once we're over the cap rather than growing unbounded.
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "3"))

_engines: "OrderedDict[str, Engine]" = OrderedDict()

def get_engine(model_name: str) -> Engine:
    if model_name in _engines:
        _engines.move_to_end(model_name)
        return _engines[model_name]

    # Loaded lazily and cached, not per-request — reloading the model on
    # every call would make each request take as long as
    # `inspect_trace.py`'s entire startup.
    engine = Engine(model_name, device_map=DEVICE_MAP, dtype=torch.float32)
    _engines[model_name] = engine
    if len(_engines) > MAX_LOADED_MODELS:
        _engines.popitem(last=False)
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine(MODEL_NAME)
    yield
    _engines.clear()


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str
    model: str = MODEL_NAME


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


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatTurn]
    model: str = MODEL_NAME
    max_new_tokens: int = 60


class PromptTemplateRequest(BaseModel):
    messages: list[ChatTurn]
    model: str = MODEL_NAME


@app.post("/api/generate")
def generate(req: GenerateRequest):
    engine = get_engine(req.model)
    return {"text": run_generate(engine, req.prompt, max_new_tokens=req.max_new_tokens)}


@app.post("/api/trace")
def trace(req: TraceRequest):
    engine = get_engine(req.model)
    return asdict(run_trace(engine, req.prompt, top_k=req.top_k))


@app.post("/api/ablate")
def ablate(req: AblateRequest):
    engine = get_engine(req.model)
    return {"text": ablate_head(engine, req.prompt, req.layer, req.head, max_new_tokens=req.max_new_tokens)}


@app.post("/api/jacobian")
def jacobian(req: JacobianRequest):
    engine = get_engine(req.model)
    return asdict(jacobian_lens(engine, req.prompt, target_token=req.target_token, top_k=req.top_k))


@app.post("/api/chat")
def chat(req: ChatRequest):
    engine = get_engine(req.model)
    return run_chat(engine, [m.model_dump() for m in req.messages], max_new_tokens=req.max_new_tokens)


@app.post("/api/prompt")
def prompt_template(req: PromptTemplateRequest):
    """Template-formats `messages` (chat template if the model has one,
    plain transcript otherwise) without generating anything — used to get
    the exact prompt for a conversation that already includes a reply, so
    it can be handed to jacobian_lens for lens analysis on the reply too.
    """
    engine = get_engine(req.model)
    return {"prompt": run_build_prompt(engine, [m.model_dump() for m in req.messages])}
