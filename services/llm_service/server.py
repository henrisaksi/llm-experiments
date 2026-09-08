import os
import sys
import gc
import threading
from typing import AsyncGenerator, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, TextIteratorStreamer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models") if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")) else "/app/models")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemma-4-E2B-it")

app = FastAPI(title="Multi-Model LLM Service")

lock = threading.Lock()
current_model_name: Optional[str] = None
model = None
processor = None
device: Optional[str] = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scan_available_models() -> list[str]:
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for entry in os.listdir(MODELS_DIR):
        entry_path = os.path.join(MODELS_DIR, entry)
        if os.path.isdir(entry_path):
            has_weights = (
                os.path.exists(os.path.join(entry_path, "model.safetensors"))
                or os.path.exists(os.path.join(entry_path, "pytorch_model.bin"))
                or any(f.endswith(".safetensors") for f in os.listdir(entry_path))
            )
            if has_weights:
                models.append(entry)
    return sorted(models)


def load_model_by_name(name: str):
    global model, processor, current_model_name, device
    model_path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' not found.")

    if device is None:
        device = get_device()

    print(f"Switching to model '{name}' on device '{device}'...")

    # Free previous model from memory
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fine-tunes/checkpoints often omit processor_config.json/preprocessor_config.json.
    # If AutoProcessor fails, fallback to the base model processor and swap in this checkpoint's tokenizer.
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception as proc_err:
        print(f"AutoProcessor.from_pretrained failed ({proc_err}), falling back to base processor with checkpoint tokenizer...")
        base_candidates = [
            os.path.join(MODELS_DIR, DEFAULT_MODEL),
            os.path.join(MODELS_DIR, "gemma-4-E2B-it"),
            "google/gemma-4-E2B-it"
        ]
        base_path = next((p for p in base_candidates if os.path.exists(p) or "/" in p), None)
        processor = AutoProcessor.from_pretrained(base_path)
        from transformers import AutoTokenizer
        processor.tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    current_model_name = name
    print(f"Model '{name}' loaded successfully.")


@app.on_event("startup")
def startup():
    available = scan_available_models()
    print(f"Available models detected: {available}")
    target = DEFAULT_MODEL if DEFAULT_MODEL in available else (available[0] if available else None)
    if target:
        with lock:
            load_model_by_name(target)
    else:
        print("Warning: No model found in models directory at startup.")


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class SwitchModelRequest(BaseModel):
    model: str


@app.get("/models")
def list_models():
    return {
        "current_model": current_model_name,
        "models": scan_available_models(),
    }


@app.post("/models/switch")
def switch_model(req: SwitchModelRequest):
    available = scan_available_models()
    if req.model not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Available: {available}"
        )
    with lock:
        if current_model_name == req.model:
            return {"status": "already_loaded", "current_model": current_model_name}
        load_model_by_name(req.model)
    return {"status": "switched", "current_model": current_model_name}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "current_model": current_model_name,
        "available_models": scan_available_models(),
        "device": device
    }


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    global current_model_name
    if req.model and req.model != current_model_name:
        available = scan_available_models()
        if req.model not in available:
            raise HTTPException(
                status_code=404,
                detail=f"Requested model '{req.model}' not found. Available: {available}"
            )
        with lock:
            if req.model != current_model_name:
                load_model_by_name(req.model)

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="No model is currently loaded")

    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = processor.apply_chat_template(
        messages_payload,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = processor(text=prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature if req.temperature > 0 else None,
        top_p=req.top_p if req.temperature > 0 else None,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def token_generator():
        for token_chunk in streamer:
            yield token_chunk

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
