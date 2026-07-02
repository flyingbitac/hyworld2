from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from hyworld2.objgen.inference import Inference, load_image


DEFAULT_CONFIG = "/models/sam-3d-objects/checkpoints/pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3D object reconstruction from an RGBA image.")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--alpha-threshold", type=int, default=0)
    return parser.parse_args()


def reconstruct_rgba(
    input_path: Path,
    output_path: Path,
    *,
    config_path: Path = Path(DEFAULT_CONFIG),
    seed: int = 42,
    compile_model: bool = False,
    alpha_threshold: int = 0,
) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM3D config not found: {config_path}")

    rgba = Image.open(input_path).convert("RGBA")
    rgba_np = np.array(rgba)
    mask = rgba_np[:, :, 3] > alpha_threshold
    if mask.sum() == 0:
        raise ValueError("Mask is empty. The input image alpha channel has no foreground pixels.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference = Inference(str(config_path), compile=compile_model)
    output = inference(load_image(str(input_path)), mask.astype("uint8"), seed=seed)
    output["gs"].save_ply(str(output_path))
    return output_path


def main() -> None:
    args = parse_args()
    output = reconstruct_rgba(
        Path(args.input).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        config_path=Path(args.config).expanduser(),
        seed=args.seed,
        compile_model=args.compile,
        alpha_threshold=args.alpha_threshold,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()

