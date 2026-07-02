from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


DEFAULT_BIREFNET_MODEL = "/models/BiRefNet"


class BiRefNetRemover:
    def __init__(self, model_path: str = DEFAULT_BIREFNET_MODEL):
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"BiRefNet model directory not found: {model_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def remove_background(self, input_path: Path, output_path: Path) -> Path:
        image = Image.open(input_path).convert("RGB")
        original_size = image.size
        model_dtype = next(self.model.parameters()).dtype
        input_tensor = self.transform(image).unsqueeze(0).to(
            device=self.device,
            dtype=model_dtype,
        )
        preds = self.model(input_tensor)[-1].sigmoid().cpu()
        mask = transforms.ToPILImage()(preds[0].squeeze())
        mask = mask.resize(original_size, Image.LANCZOS)
        rgba = image.convert("RGBA")
        rgba.putalpha(mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(output_path)
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove object image background with BiRefNet.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_BIREFNET_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remover = BiRefNetRemover(args.model)
    remover.remove_background(Path(args.input), Path(args.output))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
