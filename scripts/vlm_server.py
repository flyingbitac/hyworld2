"""Lightweight OpenAI-compatible VLM server for WorldNav (stages 1-2).

Drop-in replacement for `vllm serve` when vLLM can't run (e.g. its bundled
FlashInfer misdetects Blackwell sm_120). Serves the configured VLM via plain
`transformers` over an OpenAI-compatible /v1/chat/completions endpoint, so
traj_generate.py / vlm_utils.py's `OpenAI(...).chat.completions.create(...)`
calls work unchanged.

Run:
    CUDA_VISIBLE_DEVICES=1 conda run -n hyworld2 python scripts/vlm_server.py

Handles both image formats used by the repo:
  - OpenAI: {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
  - Qwen:   {"type":"image","image":"file:///abs/path.png"}
"""
import base64
import io
import os
import re
import sys

import torch
from PIL import Image
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

MODEL_PATH = os.environ.get("VLM_MODEL", "/models/Qwen/Qwen3.5-4B")
SERVED_NAME = os.environ.get("VLM_NAME", "Qwen/Qwen3.5-4B")
PORT = int(os.environ.get("PORT", "8000"))

if MODEL_PATH.startswith("/models/") and not os.path.isdir(MODEL_PATH):
    qwen_model_path = os.path.join("/models/Qwen", os.path.basename(MODEL_PATH))
    if os.path.isdir(qwen_model_path):
        print(f"[vlm-shim] resolved {MODEL_PATH} -> {qwen_model_path}", flush=True)
        MODEL_PATH = qwen_model_path

print(f"[vlm-shim] loading {MODEL_PATH} ...", flush=True)
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
).eval().to("cuda")
print(f"[vlm-shim] loaded on {model.device}", flush=True)

app = FastAPI()


def _apply_chat_template(conversation):
    return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        text = text.split("<think>", 1)[0].strip()
    return text


def _load_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        # data:image/png;base64,XXXX
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://"):]).convert("RGB")
    if url.startswith("http"):
        import urllib.request
        return Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
    return Image.open(url).convert("RGB")


def _parse_message_content(content):
    """Return (text, [PIL.Image]) extracted from one message's content."""
    if isinstance(content, str):
        return content, []
    text_parts, images = [], []
    for part in content:
        t = part.get("type")
        if t == "text":
            text_parts.append(part.get("text", ""))
        elif t == "image_url":
            url = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("image_url")
            images.append(_load_url(url))
        elif t == "image":
            ref = part.get("image") or part.get("image_url")
            if isinstance(ref, dict):
                ref = ref.get("url")
            images.append(_load_url(ref))
    return "\n".join(text_parts), images


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req = await request.json()
    messages = req.get("messages", [])
    all_images = []
    conversation = []
    raw_templated = False
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        text, imgs = _parse_message_content(content)
        if isinstance(content, str) and "<|im_start|>" in content:
            raw_templated = True
            conversation.append({"role": role, "content": content})
            continue
        all_images.extend(imgs)
        parts = [{"type": "text", "text": text}] + [{"type": "image"} for _ in imgs]
        conversation.append({"role": role, "content": parts})

    if raw_templated:
        prompt = "".join(m["content"] for m in conversation if isinstance(m.get("content"), str))
        inputs = processor(text=[prompt], images=all_images or None, return_tensors="pt", padding=True).to(model.device)
    else:
        prompt = _apply_chat_template(conversation)
        inputs = processor(text=[prompt], images=all_images or None, return_tensors="pt", padding=True).to(model.device)

    in_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(req.get("max_tokens", req.get("max_completion_tokens", 1024))),
            do_sample=False,
        )
    gen = out[0][in_len:]
    answer = _strip_thinking(processor.decode(gen, skip_special_tokens=True).strip())

    return {
        "object": "chat.completion",
        "model": req.get("model", SERVED_NAME),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": in_len, "completion_tokens": int(out.shape[1]) - in_len, "total_tokens": int(out.shape[1])},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.environ.get("UVICORN_LOG_LEVEL", "warning"))
