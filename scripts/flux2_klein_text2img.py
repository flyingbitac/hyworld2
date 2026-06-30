#!/usr/bin/env python3
"""Run FLUX.2 Klein text-to-image inference from a local Diffusers repo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline


MODEL_DIRS = {
    "4b": "/models/FLUX.2-klein-4B",
    "9b": "/models/FLUX.2-klein-9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX.2 Klein text-to-image runner.")
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), default="4b")
    parser.add_argument("--model-path", default=None, help="Override local model path.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for generation/offload hooks.",
    )
    parser.add_argument(
        "--placement",
        choices=("offload", "cuda"),
        default="offload",
        help="Use CPU offload for lower peak VRAM, or keep the pipeline on CUDA.",
    )
    return parser.parse_args()


def require_model_dir(model_path: Path) -> None:
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "vae/config.json",
    ]
    missing = [name for name in required if not model_path.joinpath(name).exists()]
    if missing:
        raise FileNotFoundError(f"{model_path} is missing required files: {missing}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path or MODEL_DIRS[args.model])
    require_model_dir(model_path)

    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    if args.placement == "cuda":
        pipe = pipe.to(args.device)
    else:
        pipe.enable_model_cpu_offload(device=torch.device(args.device))

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
