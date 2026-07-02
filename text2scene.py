#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import docker as hyworld_docker


REPO_ROOT = Path(__file__).resolve().parent
CONTAINER_WORKDIR = "/workspace/hyworld2"
DEFAULT_CONTAINER = "hyworld2-base"
DEFAULT_MODEL_PATH = "/models/Qwen/Qwen3.5-4B"
DEFAULT_SAM3D_CONFIG = "/models/sam-3d-objects/checkpoints/pipeline.yaml"
DEFAULT_BIREFNET_MODEL = "/models/BiRefNet"
CONTAINER_CACHE_EXPORT = (
    "export HF_MODULES_CACHE=/tmp/hyworld2_hf_modules "
    "TRANSFORMERS_CACHE=/tmp/hyworld2_transformers_cache "
    "DINO_MODEL=/models/dinov2-with-registers-large; "
)
VALID_STAGES = {"parse", "worldgen", "object-image", "objgen", "calibrate"}
RUNNAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


SYSTEM_PROMPT = """
You are a 3D scene text parser.

Your task is to extract a structured 3D scene JSON from the user's input.

IMPORTANT:
- The user input may be Chinese, English, or mixed.
- You MUST translate all scene names, object names, position hints, and size hints into English.
- The output JSON must contain English text only.
- Output ONLY valid JSON.
- Do NOT output Markdown code blocks.
- Do NOT explain anything.

Required JSON schema:
{
  "scene": "English scene name",
  "obstacles": [
    {
      "name": "English object name",
      "count": 1,
      "position_hint": "English position hint or null",
      "size_hint": "English size hint or null",
      "real_size": {
        "length_m": 1.0,
        "width_m": 1.0,
        "height_m": 1.0
      }
    }
  ]
}

Rules:
- scene must be one English string.
- obstacles must be an array.
- count must be an integer >= 1.
- If quantity is not specified, use count = 1.
- If position is not specified, use position_hint = null.
- If size text is not specified, use size_hint = null.
- real_size must always be provided and cannot be null.
- real_size values must be numbers only, without unit strings.
- Use meters for all dimensions.
- Use concrete English object names, not vague translations.
- For plants, distinguish between "potted plant", "large indoor plant", "tree", "grass lawn", and "shrub".
- Avoid humanoid/object ambiguity in names.

Reference sizes:
- potted plant: about 0.4 x 0.4 x 1.0
- small potted plant: about 0.25 x 0.25 x 0.4
- tall tree: about 1.5 x 1.5 x 8.0
- treadmill: about 1.8 x 0.8 x 1.4
- chair: about 0.5 x 0.5 x 0.9
- table: about 1.2 x 0.7 x 0.75
- sofa: about 2.0 x 0.9 x 0.85
- bed: about 2.0 x 1.5 x 0.6
- cabinet: about 0.8 x 0.5 x 1.8
- TV: about 1.2 x 0.1 x 0.7
"""


def require_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    return data


def positive_float(data: dict[str, Any], key: str) -> float:
    try:
        value = float(data[key])
    except KeyError as exc:
        raise ValueError(f"missing required field: {key}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text fields must be strings or null")
    return value


def validate_scene_config(data: Any) -> dict[str, Any]:
    data = require_mapping(data, "scene config")
    scene = data.get("scene")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("scene must be a non-empty string")
    raw_obstacles = data.get("obstacles")
    if not isinstance(raw_obstacles, list):
        raise ValueError("obstacles must be an array")

    obstacles = []
    for item in raw_obstacles:
        item = require_mapping(item, "obstacle")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("obstacle.name must be a non-empty string")
        try:
            count = int(item.get("count", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("obstacle.count must be an integer") from exc
        if count < 1:
            raise ValueError("obstacle.count must be >= 1")
        real_size = require_mapping(item.get("real_size"), "real_size")
        obstacles.append(
            {
                "name": name.strip(),
                "count": count,
                "position_hint": optional_str(item.get("position_hint")),
                "size_hint": optional_str(item.get("size_hint")),
                "real_size": {
                    "length_m": positive_float(real_size, "length_m"),
                    "width_m": positive_float(real_size, "width_m"),
                    "height_m": positive_float(real_size, "height_m"),
                },
            }
        )

    return {"scene": scene.strip(), "obstacles": obstacles}


def extract_json(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"model output did not contain a JSON object: {text}")
    return match.group(0)


def load_scene_config(path: Path | str) -> dict[str, Any]:
    return validate_scene_config(json.loads(Path(path).read_text(encoding="utf-8")))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_runname(runname: str) -> str:
    if not RUNNAME_RE.fullmatch(runname):
        raise ValueError("--runname may only contain letters, numbers, dot, underscore, and dash.")
    return runname


def safe_slug(text: str, *, fallback: str = "item", max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not slug:
        slug = fallback
    return slug[:max_length].rstrip("-") or fallback


def object_dir_name(index: int, name: str) -> str:
    return f"{index:03d}_{safe_slug(name, fallback='object')}"


def build_scene_prompt(original_prompt: str, scene_config: dict[str, Any] | None) -> str:
    if scene_config is None:
        return original_prompt
    object_desc = []
    for obstacle in scene_config["obstacles"]:
        phrase = f"{obstacle['count']} {obstacle['name']}"
        if obstacle.get("position_hint"):
            phrase += f" placed {obstacle['position_hint']}"
        object_desc.append(phrase)
    if not object_desc:
        return f"{original_prompt}. Scene type: {scene_config['scene']}."
    return (
        f"{original_prompt}. Scene type: {scene_config['scene']}. "
        f"Must contain: {', '.join(object_desc)}. No people."
    )


def build_object_prompt(obstacle: dict[str, Any]) -> str:
    prompt = (
        f"commercial product photo of one {obstacle['name']}, "
        "single standalone object only, full object visible, centered object, "
        "isolated on plain white background, studio lighting, realistic, high detail, "
        "catalog image, ecommerce product image, no people, no text, no watermark"
    )
    if obstacle.get("size_hint"):
        prompt += f", {obstacle['size_hint']}"
    return prompt


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

    def parse(self, text: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
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


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def host_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def container_path(path: str | Path) -> str:
    host = Path(path)
    if not host.is_absolute():
        host = REPO_ROOT / host
    relative = host.resolve().relative_to(REPO_ROOT.resolve())
    return f"{CONTAINER_WORKDIR}/{relative.as_posix()}"


def container_to_host_model_path(model_path: str) -> Path | None:
    if not model_path.startswith("/models/"):
        return None
    return REPO_ROOT / "models" / model_path[len("/models/"):]


def primary_device(device: str) -> str:
    return device.split(",", 1)[0].strip() or "0"


def parse_skip(spec: str) -> frozenset[str]:
    if not spec.strip():
        return frozenset()
    values = {item.strip() for item in spec.split(",") if item.strip()}
    unknown = values - VALID_STAGES
    if unknown:
        raise ValueError(f"unknown --skip stages: {', '.join(sorted(unknown))}")
    return frozenset(values)


def build_manifest(
    prompt: str,
    runname: str,
    scene_config_path: Path,
    worldgen_dir: Path,
    scene_config: dict[str, Any],
) -> dict[str, Any]:
    text2scene_dir = scene_config_path.parent
    objects = []
    for index, obstacle in enumerate(scene_config["obstacles"]):
        object_dir = text2scene_dir / "objects" / object_dir_name(index, obstacle["name"])
        objects.append(
            {
                "index": index,
                "name": obstacle["name"],
                "count": obstacle["count"],
                "raw_image": repo_relative(object_dir / "raw.png"),
                "rgba_image": repo_relative(object_dir / "rgba.png"),
                "raw_ply": repo_relative(object_dir / "raw.ply"),
                "calibrated_ply": repo_relative(object_dir / "calibrated.ply"),
                "status": "pending",
            }
        )
    return {
        "prompt": prompt,
        "runname": runname,
        "scene_config": repo_relative(scene_config_path),
        "worldgen": {
            "run_dir": repo_relative(worldgen_dir),
            "panorama": repo_relative(worldgen_dir / "panorama.png"),
            "gs_results": repo_relative(worldgen_dir / "gs_results"),
        },
        "objects": objects,
    }


def set_asset_status(asset: dict[str, Any], status: str, error: str | None = None) -> None:
    asset["status"] = status
    if error:
        asset["error"] = error
    else:
        asset.pop("error", None)


def worldgen_command(args: argparse.Namespace, scene_prompt: str) -> list[str]:
    command = [
        sys.executable,
        "docker.py",
        "--name",
        args.container_name,
        "run",
        "--prompt",
        scene_prompt,
        "--runname",
        args.runname,
        "--device",
        args.device,
        "--panorama-backend",
        args.panorama_backend,
        "--seed",
        str(args.seed),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def run_host(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_container(container_name: str, command: str, *, capture: bool = False) -> str:
    docker_command = ["docker", "exec", container_name, "bash", "-lc", CONTAINER_CACHE_EXPORT + command]
    print("+", shlex.join(docker_command), flush=True)
    if capture:
        result = subprocess.run(
            docker_command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stderr:
            print(result.stderr, end="")
        if result.stdout:
            print(result.stdout, end="")
        return result.stdout
    subprocess.run(docker_command, cwd=REPO_ROOT, check=True)
    return ""


def chown_container_path(container_name: str, path: str | Path) -> None:
    uid = os.getuid()
    gid = os.getgid()
    target = container_path(path)
    run_container(
        container_name,
        f"if [ -e {shlex.quote(target)} ]; then chown -R {uid}:{gid} {shlex.quote(target)}; fi",
    )


def ensure_container(args: argparse.Namespace) -> None:
    if hyworld_docker.container_running(args.container_name):
        return
    start_args = SimpleNamespace(
        image=hyworld_docker.DEFAULT_IMAGE,
        name=args.container_name,
        build=False,
        models=hyworld_docker.DEFAULT_MODELS,
        writable_models=False,
        display=False,
        user=None,
        flash_attn=True,
        worldgen_extras=True,
    )
    hyworld_docker.start(start_args)


def print_dry_run(args: argparse.Namespace, text2scene_dir: Path, scene_config_path: Path, manifest_path: Path, worldgen_dir: Path) -> None:
    print(f"text2scene_dir={text2scene_dir}")
    print(f"scene_config={scene_config_path}")
    print(f"manifest={manifest_path}")
    print(f"worldgen_dir={worldgen_dir}")
    print("+", shlex.join(worldgen_command(args, args.prompt)))


def run_parse_stage(args: argparse.Namespace, scene_config_path: Path) -> dict[str, Any]:
    if "parse" in args.skip:
        return load_scene_config(scene_config_path)
    if args.skip_existing and scene_config_path.is_file():
        return load_scene_config(scene_config_path)

    ensure_container(args)
    run_container(
        args.container_name,
        "cd /workspace/hyworld2 && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device(args.device))} "
        "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
        "python -u -m hyworld2.objgen.parse_scene "
        f"--prompt {shlex.quote(args.prompt)} "
        f"--output {shlex.quote(container_path(scene_config_path))} "
        f"--model {shlex.quote(args.parser_model)}",
    )
    chown_container_path(args.container_name, scene_config_path.parent)
    return load_scene_config(scene_config_path)


def run_object_image_stage(args: argparse.Namespace, scene_config: dict[str, Any], manifest: dict[str, Any]) -> None:
    ensure_container(args)
    device = primary_device(args.device)
    for asset, obstacle in zip(manifest["objects"], scene_config["obstacles"]):
        if args.skip_existing and host_path(asset["rgba_image"]).is_file():
            set_asset_status(asset, "image_generated")
            continue
        raw_image = container_path(asset["raw_image"])
        rgba_image = container_path(asset["rgba_image"])
        try:
            run_container(
                args.container_name,
                "cd /workspace/hyworld2 && "
                f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} "
                "/opt/miniconda3/bin/conda run --no-capture-output -n flux2 "
                "python -u scripts/flux2_klein_text2img.py "
                "--model 9b --model-path /models/FLUX.2-klein-9B "
                f"--prompt {shlex.quote(build_object_prompt(obstacle))} "
                f"--output {shlex.quote(raw_image)} "
                "--height 1024 --width 1024 --steps 4 --guidance-scale 1.0 "
                f"--seed {args.seed + asset['index'] + 1} --placement offload",
            )
            run_container(
                args.container_name,
                "cd /workspace/hyworld2 && "
                f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} "
                "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                "python -u -m hyworld2.objgen.object_image "
                f"--input {shlex.quote(raw_image)} "
                f"--output {shlex.quote(rgba_image)} "
                f"--model {shlex.quote(args.birefnet_model)}",
            )
            set_asset_status(asset, "image_generated")
            asset.pop("rgba_mode", None)
            asset.pop("warning", None)
        except subprocess.CalledProcessError as exc:
            set_asset_status(asset, "image_failed", str(exc))
        finally:
            chown_container_path(args.container_name, host_path(asset["raw_image"]).parent)


def run_objgen_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    ensure_container(args)
    device = primary_device(args.device)
    for asset in manifest["objects"]:
        if asset["status"] not in {"image_generated", "object_reconstructed"}:
            if host_path(asset["rgba_image"]).is_file():
                set_asset_status(asset, "image_generated")
            else:
                continue
        if args.skip_existing and host_path(asset["raw_ply"]).is_file():
            set_asset_status(asset, "object_reconstructed")
            continue
        try:
            run_container(
                args.container_name,
                "cd /workspace/hyworld2 && "
                f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} "
                "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                "python -u -m hyworld2.objgen.run_from_rgba "
                f"--input {shlex.quote(container_path(asset['rgba_image']))} "
                f"--output {shlex.quote(container_path(asset['raw_ply']))} "
                f"--config {shlex.quote(args.sam3d_config)} "
                f"--seed {args.seed + asset['index'] + 1}",
            )
            set_asset_status(asset, "object_reconstructed")
        except subprocess.CalledProcessError as exc:
            set_asset_status(asset, "objgen_failed", str(exc))
        finally:
            chown_container_path(args.container_name, host_path(asset["raw_ply"]).parent)


def run_calibrate_stage(args: argparse.Namespace, scene_config_path: Path, manifest: dict[str, Any]) -> None:
    ensure_container(args)
    for asset in manifest["objects"]:
        if asset["status"] not in {"object_reconstructed", "calibrated"}:
            if host_path(asset["raw_ply"]).is_file():
                set_asset_status(asset, "object_reconstructed")
            else:
                continue
        if args.skip_existing and host_path(asset["calibrated_ply"]).is_file():
            set_asset_status(asset, "calibrated")
            continue
        command = (
            "cd /workspace/hyworld2 && "
            "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
            "python -u -m hyworld2.objgen.calibrate_ply "
            f"--input {shlex.quote(container_path(asset['raw_ply']))} "
            f"--output {shlex.quote(container_path(asset['calibrated_ply']))} "
            f"--scene-config {shlex.quote(container_path(scene_config_path))} "
            f"--index {asset['index']}"
        )
        try:
            output = run_container(args.container_name, command, capture=True)
            asset["calibration"] = json.loads(extract_json(output))
            set_asset_status(asset, "calibrated")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            set_asset_status(asset, "calibrate_failed", str(exc))
        finally:
            chown_container_path(args.container_name, host_path(asset["calibrated_ply"]).parent)


def run_workflow(args: argparse.Namespace) -> dict[str, Any] | None:
    validate_runname(args.runname)
    text2scene_dir = REPO_ROOT / "examples" / "text2scene" / args.runname
    scene_config_path = text2scene_dir / "scene_config.json"
    manifest_path = text2scene_dir / "manifest.json"
    worldgen_dir = REPO_ROOT / "examples" / "worldgen" / args.runname

    if args.dry_run:
        print_dry_run(args, text2scene_dir, scene_config_path, manifest_path, worldgen_dir)
        return None

    scene_config = run_parse_stage(args, scene_config_path)
    manifest = build_manifest(args.prompt, args.runname, scene_config_path, worldgen_dir, scene_config)
    save_json(manifest_path, manifest)

    if "worldgen" not in args.skip:
        run_host(worldgen_command(args, build_scene_prompt(args.prompt, scene_config)))

    if "object-image" not in args.skip:
        run_object_image_stage(args, scene_config, manifest)
        save_json(manifest_path, manifest)

    if "objgen" not in args.skip:
        run_objgen_stage(args, manifest)
        save_json(manifest_path, manifest)

    if "calibrate" not in args.skip:
        run_calibrate_stage(args, scene_config_path, manifest)
        save_json(manifest_path, manifest)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text prompt -> HY-World scene + SAM3D object assets.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--runname", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--panorama-backend", choices=("hypano", "flux-lora"), default="hypano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parser-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sam3d-config", default=DEFAULT_SAM3D_CONFIG)
    parser.add_argument("--birefnet-model", default=DEFAULT_BIREFNET_MODEL)
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER)
    args = parser.parse_args()
    args.skip = parse_skip(args.skip)
    return args


def main() -> None:
    args = parse_args()
    manifest = run_workflow(args)
    if manifest is not None:
        print(f"manifest: examples/text2scene/{args.runname}/manifest.json")


if __name__ == "__main__":
    main()
