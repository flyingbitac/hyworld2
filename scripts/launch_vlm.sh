#!/bin/bash
# Launch the OpenAI-compatible VLM shim for WorldNav (stages 1-2).
# This is the chosen VLM server (vLLM 0.23 can't run on Blackwell: its bundled
# FlashInfer misdetects sm_120). The shim serves Qwen3.5-4B via plain transformers
# and exposes the same OpenAI /v1/chat/completions endpoint that
# traj_generate.py expects.
set -e
export HOME=/cache/torch/vlm-shim-home
export HF_HOME=/models/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PORT=${PORT:-8000}
mkdir -p "$HOME"

exec /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
    python -u /workspace/hyworld2/scripts/vlm_server.py
