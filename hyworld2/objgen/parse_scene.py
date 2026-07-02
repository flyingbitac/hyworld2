from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2scene import DEFAULT_MODEL_PATH, SYSTEM_PROMPT, extract_json, save_json, validate_scene_config


class SceneParser:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self._torch = torch
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_classes = [AutoModelForCausalLM]
        model_type = getattr(config, "model_type", "")
        architectures = getattr(config, "architectures", None) or []
        if "vl" in model_type.lower() or any("vl" in item.lower() for item in architectures):
            model_classes = []
            try:
                from transformers import Qwen3VLForConditionalGeneration

                model_classes.append(Qwen3VLForConditionalGeneration)
            except ImportError:
                pass
            try:
                from transformers import AutoModelForImageTextToText

                model_classes.append(AutoModelForImageTextToText)
            except ImportError:
                pass
            model_classes.append(AutoModelForCausalLM)

        last_error: Exception | None = None
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(f"failed to load parser model from {model_path}") from last_error

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except (OSError, ValueError):
            self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if getattr(self.processor, "pad_token_id", None) is None and getattr(self.processor, "eos_token", None):
            self.processor.pad_token = self.processor.eos_token

        self.generation_kwargs = {"max_new_tokens": 1024, "do_sample": False}
        pad_token_id = getattr(self.processor, "pad_token_id", None)
        if pad_token_id is not None:
            self.generation_kwargs["pad_token_id"] = pad_token_id

    def parse(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(self.model.parameters()).device
        inputs = self.processor(text=[prompt], return_tensors="pt").to(device)

        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.generation_kwargs)

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return validate_scene_config(json.loads(extract_json(output_text)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse text prompt into text2scene scene_config.json.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_config = SceneParser(args.model).parse(args.prompt)
    save_json(Path(args.output), scene_config)
    print(json.dumps(scene_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
