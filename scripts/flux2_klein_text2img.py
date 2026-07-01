#!/usr/bin/env python3
"""Run FLUX.2 Klein text-to-image inference from a local Diffusers repo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


MODEL_DIRS = {
    "4b": "/models/FLUX.2-klein-4B",
    "9b": "/models/FLUX.2-klein-9B",
}
PANORAMA_PROMPT_PREFIX = "Equirectangular 360 panorama, "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX.2 Klein text-to-image runner.")
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), default="4b")
    parser.add_argument("--model-path", default=None, help="Override local model path.")
    parser.add_argument("--lora-path", default=None, help="Optional Diffusers LoRA path.")
    parser.add_argument("--lora-weight-name", default=None, help="Optional LoRA safetensors filename.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA adapter scale.")
    parser.add_argument(
        "--panorama-trigger",
        action="store_true",
        help=f"Ensure the prompt starts with '{PANORAMA_PROMPT_PREFIX}'.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--circular-blend-width",
        type=int,
        default=0,
        help="Blend and crop this many pixels at the panorama seam. 0 disables it.",
    )
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


def circular_blend_edges(image: Image.Image, blend_width: int) -> Image.Image:
    if blend_width <= 0:
        return image

    arr = np.array(image.convert("RGB"))
    if blend_width >= arr.shape[1]:
        raise ValueError(f"--circular-blend-width must be smaller than image width ({arr.shape[1]}).")

    for x in range(blend_width):
        arr[:, x, :] = (
            arr[:, -blend_width + x, :] * (1 - x / blend_width)
            + arr[:, x, :] * (x / blend_width)
        )
    return Image.fromarray(arr[:, :-blend_width].astype(np.uint8))


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
    if args.lora_path:
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Loading FLUX LoRA weights requires `peft` in the flux2 environment. "
                "Install it with: conda run -n flux2 python -m pip install peft"
            ) from exc
        pipe.load_lora_weights(
            args.lora_path,
            weight_name=args.lora_weight_name,
            adapter_name="panorama",
        )
        pipe.set_adapters(["panorama"], adapter_weights=[args.lora_scale])

    if args.placement == "cuda":
        pipe = pipe.to(args.device)
    else:
        pipe.enable_model_cpu_offload(device=torch.device(args.device))

    prompt = args.prompt
    if args.panorama_trigger and not prompt.lower().startswith(PANORAMA_PROMPT_PREFIX.lower()):
        prompt = f"{PANORAMA_PROMPT_PREFIX}{prompt}"

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    image = circular_blend_edges(image, args.circular_blend_width)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
