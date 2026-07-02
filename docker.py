#!/usr/bin/env python3
"""Small Docker interface for the HY-World container images."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = "hyworld2-base:v1.0"
DEFAULT_CONTAINER = "hyworld2-base"
DEFAULT_DOCKERFILE = Path("Dockerfile")
DEFAULT_MODELS = REPO_ROOT / "models"
DEFAULT_VLM = "Qwen/Qwen3.5-4B"
ALIYUN_IMAGE = "crpi-jq3nu6qbricb9zcb.cn-beijing.personal.cr.aliyuncs.com/zxh_in_bitac/hyworld2"
CONTAINER_WORKDIR = "/workspace/hyworld2"
CONTAINER_MODELS = "/models"
DEFAULT_VLM_PATH = f"{CONTAINER_MODELS}/{DEFAULT_VLM}"
CONDA = "/opt/miniconda3/bin/conda"
FLUX_PANORAMA_LORA_REPO = "crafiq/flux-2-klein-9b-360-panorama-lora"
FLUX_PANORAMA_LORA_DIR = "flux-2-klein-9b-360-panorama-lora"
FLUX_PANORAMA_LORA_WEIGHT = "flux-2-klein-9b-360-panorama-lora.safetensors"
MODEL_DOWNLOADS = (
    ("black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-9B", "FLUX.2-klein-9B", ()),
    (
        FLUX_PANORAMA_LORA_REPO,
        FLUX_PANORAMA_LORA_REPO,
        FLUX_PANORAMA_LORA_DIR,
        (FLUX_PANORAMA_LORA_WEIGHT,),
    ),
    ("Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509", ()),
    (
        "Tencent-Hunyuan/HY-World-2.0",
        "tencent/HY-World-2.0",
        "HY-World-2.0",
        ("HY-Pano-2.0/pytorch_lora_weights.safetensors", "HY-WorldMirror-2.0/*"),
    ),
    (DEFAULT_VLM, DEFAULT_VLM, DEFAULT_VLM, ()),
    ("facebook/sam3", "facebook/sam3", "sam3", ()),
    ("facebook/sam-3d-objects", "facebook/sam-3d-objects", "sam-3d-objects", ()),
    ("hanshanxue/WorldStereo", "hanshanxue/WorldStereo", "WorldStereo", ("worldstereo-memory-dmd/*",)),
    (
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "Wan2.1-I2V-14B-480P-Diffusers",
        (),
    ),
    ("Ruicheng/moge-2-vitl-normal", "Ruicheng/moge-2-vitl-normal", "moge-2-vitl-normal", ()),
    ("naver-iv/zim-anything-vitl", "naver-iv/zim-anything-vitl", "zim-anything-vitl", ()),
    ("IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny", "grounding-dino-tiny", ()),
    ("facebook/dinov2-base", "facebook/dinov2-base", "dinov2-base", ()),
    (
        "facebook/dinov2-with-registers-large",
        "facebook/dinov2-with-registers-large",
        "dinov2-with-registers-large",
        (),
    ),
    ("AI-ModelScope/ZhengPeng7-BiRefNet", "ZhengPeng7/BiRefNet", "BiRefNet", ()),
)
HY_WORLD_REQUIRED_FILES = (
    "HY-Pano-2.0/pytorch_lora_weights.safetensors",
    "HY-WorldMirror-2.0/config.json",
    "HY-WorldMirror-2.0/model.safetensors",
)
MODEL_REQUIRED_FILES = {
    "FLUX.2-klein-9B": ("model_index.json",),
    FLUX_PANORAMA_LORA_DIR: (FLUX_PANORAMA_LORA_WEIGHT,),
    "Qwen/Qwen-Image-Edit-2509": ("model_index.json",),
    "HY-World-2.0": HY_WORLD_REQUIRED_FILES,
    DEFAULT_VLM: ("config.json",),
    "sam3": ("config.json",),
    "sam-3d-objects": (
        "configuration.json",
        "checkpoints/pipeline.yaml",
        "checkpoints/ss_generator.yaml",
        "checkpoints/ss_generator.ckpt",
        "checkpoints/slat_generator.yaml",
        "checkpoints/slat_generator.ckpt",
        "checkpoints/ss_decoder.yaml",
        "checkpoints/ss_decoder.ckpt",
        "checkpoints/ss_encoder.yaml",
        "checkpoints/ss_encoder.ckpt",
        "checkpoints/ss_encoder.safetensors",
        "checkpoints/slat_decoder_gs.yaml",
        "checkpoints/slat_decoder_gs.ckpt",
        "checkpoints/slat_decoder_gs_4.yaml",
        "checkpoints/slat_decoder_gs_4.ckpt",
        "checkpoints/slat_decoder_mesh.yaml",
        "checkpoints/slat_decoder_mesh.ckpt",
        "checkpoints/slat_decoder_mesh.pt",
        "checkpoints/slat_encoder.yaml",
        "checkpoints/slat_encoder.ckpt",
    ),
    "WorldStereo": ("worldstereo-memory-dmd/config.json", "worldstereo-memory-dmd/model.safetensors"),
    "Wan2.1-I2V-14B-480P-Diffusers": ("model_index.json",),
    "moge-2-vitl-normal": ("model.pt",),
    "zim-anything-vitl": ("zim_vit_l_2092/encoder.onnx", "zim_vit_l_2092/decoder.onnx"),
    "grounding-dino-tiny": (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "dinov2-base": ("config.json", "preprocessor_config.json", "model.safetensors"),
    "dinov2-with-registers-large": ("config.json", "preprocessor_config.json", "model.safetensors"),
    "BiRefNet": ("config.json",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def stage_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug[:72] or "stage"


def format_seconds(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}s"


class ProfileRecorder:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        scene: str,
        devices: list[str],
        stage3_offload_mode: str,
        skip_stages: set[int],
    ) -> None:
        self.runname = args.runname
        self.devices = devices
        self.sample_interval = args.profile_interval
        self.container_dir = f"{scene}/profiles"
        self.host_dir = REPO_ROOT / "examples" / "worldgen" / args.runname / "profiles"
        self.host_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.started_at_iso = utc_now()
        self.stage_index = 0
        self.stages: list[dict[str, object]] = []
        self.run_metadata = {
            "image": args.image,
            "container": args.name,
            "prompt": args.prompt,
            "panorama_backend": args.panorama_backend,
            "vlm": DEFAULT_VLM,
            "device": args.device,
            "stage3_offload_mode": stage3_offload_mode,
            "skip": sorted(skip_stages),
            "skip_existing": args.skip_existing,
            "batchsize": args.batchsize,
            "gs_steps": args.gs_steps,
        }

    @property
    def gpu_filter(self) -> str:
        return ",".join(self.devices)

    def wrap_stage(self, label: str, command: str) -> tuple[dict[str, object], str]:
        index = self.stage_index
        self.stage_index += 1
        slug = stage_slug(label)
        stem = f"stage_{index:02d}_{slug}"
        host_csv = self.host_dir / f"{stem}.csv"
        host_summary = self.host_dir / f"{stem}.json"
        container_csv = f"{self.container_dir}/{stem}.csv"
        container_summary = f"{self.container_dir}/{stem}.json"
        record: dict[str, object] = {
            "index": index,
            "label": label,
            "slug": slug,
            "status": "running",
            "skipped": False,
            "csv": repo_relative(host_csv),
            "summary": repo_relative(host_summary),
        }
        self.stages.append(record)

        inner = f"set -euo pipefail; {command}"
        profiled_command = (
            f"cd {shlex.quote(CONTAINER_WORKDIR)} && "
            "/opt/miniconda3/bin/python scripts/profile_gpu.py "
            f"--sample-interval {shlex.quote(str(self.sample_interval))} "
            f"--gpus {shlex.quote(self.gpu_filter)} "
            f"--csv {shlex.quote(container_csv)} "
            f"--summary {shlex.quote(container_summary)} "
            f"-- bash -lc {shlex.quote(inner)}"
        )
        return record, profiled_command

    def finish_stage(self, record: dict[str, object]) -> None:
        summary_path = REPO_ROOT / str(record["summary"])
        if not summary_path.is_file():
            record["status"] = "summary-missing"
            record["summary_missing"] = True
            return
        summary = json.loads(summary_path.read_text())
        returncode = summary.get("returncode")
        record["returncode"] = returncode
        record["elapsed_sec"] = summary.get("elapsed_sec")
        record["gpus"] = summary.get("gpus", {})
        if summary.get("interrupted"):
            record["interrupted"] = True
        record["status"] = "ok" if returncode == 0 else "failed"

    def record_skipped(self, stage_number: int, label: str) -> None:
        index = self.stage_index
        self.stage_index += 1
        full_label = f"Skip Stage {stage_number}: {label}"
        self.stages.append(
            {
                "index": index,
                "label": full_label,
                "slug": stage_slug(full_label),
                "status": "skipped",
                "skipped": True,
                "elapsed_sec": 0.0,
                "gpus": {},
            }
        )

    def write(self, *, status: str, error: str | None = None) -> None:
        for record in self.stages:
            if not record.get("skipped") and record.get("status") == "running":
                self.finish_stage(record)

        elapsed = time.time() - self.started_at
        profile = {
            "runname": self.runname,
            "status": status,
            "error": error,
            "started_at": self.started_at_iso,
            "ended_at": utc_now(),
            "elapsed_sec": round(elapsed, 3),
            "devices": self.devices,
            "gpus_filter": [int(device) for device in self.devices],
            "sample_interval_sec": self.sample_interval,
            "profile_dir": repo_relative(self.host_dir),
            "run": self.run_metadata,
            "stages": self.stages,
        }
        profile_json = self.host_dir / "profile.json"
        profile_md = self.host_dir / "profile.md"
        profile_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
        profile_md.write_text(self.render_markdown(profile))
        print(f"[PROFILE] summary: {profile_json}", flush=True)
        print(f"[PROFILE] report:  {profile_md}", flush=True)

    def render_markdown(self, profile: dict[str, object]) -> str:
        lines = [
            f"# HY-World run profile: {self.runname}",
            "",
            f"- Status: {profile['status']}",
            f"- Elapsed: {format_seconds(profile['elapsed_sec'])}",
            f"- Devices: {','.join(self.devices)}",
            f"- Sample interval: {self.sample_interval}s",
        ]
        if profile.get("error"):
            lines.append(f"- Error: `{profile['error']}`")
        lines.extend(
            [
                "",
                "| # | Stage | Status | Return code | Elapsed | Peak memory MiB | Avg GPU util % | Files |",
                "|---|-------|--------|-------------|---------|-----------------|----------------|-------|",
            ]
        )
        for record in self.stages:
            gpus = record.get("gpus", {})
            peak = "-"
            util = "-"
            if isinstance(gpus, dict) and gpus:
                peak = ", ".join(
                    f"{gpu}: {stats.get('peak_memory_mib', '-')}"
                    for gpu, stats in sorted(gpus.items(), key=lambda item: int(item[0]))
                    if isinstance(stats, dict)
                )
                util = ", ".join(
                    f"{gpu}: {stats.get('avg_utilization_gpu_pct', '-')}"
                    for gpu, stats in sorted(gpus.items(), key=lambda item: int(item[0]))
                    if isinstance(stats, dict)
                )
            files = "-"
            if record.get("csv") and record.get("summary"):
                files = f"{record['csv']}, {record['summary']}"
            lines.append(
                "| {index} | {label} | {status} | {returncode} | {elapsed} | {peak} | {util} | {files} |".format(
                    index=record["index"],
                    label=str(record["label"]).replace("|", "\\|"),
                    status=record.get("status", "-"),
                    returncode=record.get("returncode", "-"),
                    elapsed=format_seconds(record.get("elapsed_sec") if isinstance(record.get("elapsed_sec"), (int, float)) else None),
                    peak=peak,
                    util=util,
                    files=files,
                )
            )
        return "\n".join(lines) + "\n"


def shell_join(*parts: str) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run(command: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env_prefix: list[str] = []
    if env and env.get("HF_ENDPOINT") != os.environ.get("HF_ENDPOINT"):
        env_prefix.append(f"HF_ENDPOINT={env['HF_ENDPOINT']}")
    print("+", shlex.join([*env_prefix, *command]), flush=True)
    return subprocess.run(command, check=check, env=env)


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()


def cli_command(names: tuple[str, ...], dry_run: bool, install_hint: str) -> list[str]:
    for name in names:
        conda_env_command = Path(sys.executable).with_name(name)
        if conda_env_command.is_file() and os.access(conda_env_command, os.X_OK):
            return [str(conda_env_command)]

        command = shutil.which(name)
        if command:
            return [command]

    if dry_run:
        return [names[0]]

    print(install_hint, file=sys.stderr)
    raise SystemExit(1)


def modelscope_command(dry_run: bool) -> list[str]:
    return cli_command(
        ("modelscope",),
        dry_run,
        "`modelscope` command was not found. Run this from an environment that has "
        "ModelScope installed, for example: `conda activate torch`.",
    )


def huggingface_download_command(dry_run: bool) -> list[str]:
    return cli_command(
        ("hf", "huggingface-cli"),
        dry_run,
        "`hf` command was not found. Install Hugging Face Hub first, "
        "for example: `python -m pip install huggingface_hub`.",
    )


def direct_hf_download_required_files(hf_id: str, local_dir: Path, required_files: tuple[str, ...], endpoint: str) -> None:
    endpoint = endpoint.rstrip("/")
    for relative_path in required_files:
        target = local_dir / relative_path
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        quoted_path = "/".join(urllib.parse.quote(part) for part in relative_path.split("/"))
        url = f"{endpoint}/{hf_id}/resolve/main/{quoted_path}"
        tmp_target = target.with_name(target.name + ".part")
        print("+", shlex.join(["HF_ENDPOINT=" + endpoint, "direct-download", url, str(target)]), flush=True)
        with urllib.request.urlopen(url) as response, tmp_target.open("wb") as output:
            shutil.copyfileobj(response, output)
        tmp_target.replace(target)


def docker_available() -> None:
    if not capture(["docker", "version", "--format", "{{.Server.Version}}"]):
        raise RuntimeError("Docker is not available or the current user cannot access the Docker daemon.")


def image_exists(image: str) -> bool:
    return bool(capture(["docker", "image", "inspect", image, "--format", "{{.Id}}"]))


def image_tag(image: str) -> str:
    image_name = image.rsplit("/", 1)[-1]
    if ":" not in image_name:
        return "latest"
    return image_name.rsplit(":", 1)[1]


def container_running(name: str) -> bool:
    names = capture(["docker", "ps", "--format", "{{.Names}}"]).splitlines()
    return name in names


def container_exists(name: str) -> bool:
    names = capture(["docker", "ps", "-a", "--format", "{{.Names}}"]).splitlines()
    return name in names


def cache_root() -> Path:
    root = Path(os.environ.get("HYWORLD_DOCKER_CACHE", "~/.cache/hyworld2-docker")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o777)
    subdirs = [
        "hf",
        "torch",
        "matplotlib",
    ]
    for subdir in subdirs:
        path = root.joinpath(subdir)
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
    return root


def mount_args(models: Path, writable_models: bool) -> list[str]:
    root = cache_root()
    mounts = [
        (REPO_ROOT, CONTAINER_WORKDIR, False),
        (models.expanduser(), CONTAINER_MODELS, not writable_models),
        (root / "hf", "/cache/huggingface", False),
        (root / "torch", "/cache/torch", False),
        (root / "matplotlib", "/cache/matplotlib", False),
    ]
    args: list[str] = []
    for source, target, read_only in mounts:
        source = Path(source).expanduser()
        if not source.exists() and target == CONTAINER_MODELS:
            raise FileNotFoundError(f"Model directory does not exist: {source}")
        if not source.exists():
            source.mkdir(parents=True, exist_ok=True)
        spec = f"type=bind,source={source.resolve()},target={target}"
        if read_only:
            spec += ",readonly"
        args.extend(["--mount", spec])
    return args


def build(args: argparse.Namespace) -> None:
    docker_available()
    dockerfile = args.dockerfile.expanduser()
    if not dockerfile.is_absolute():
        dockerfile = REPO_ROOT / dockerfile
    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile does not exist: {dockerfile}")
    command = [
        "docker",
        "build",
        "--progress=plain",
        "--file",
        str(dockerfile),
        "--tag",
        args.image,
        "--build-arg",
        f"INSTALL_FLASH_ATTN={int(args.flash_attn)}",
        "--build-arg",
        f"INSTALL_WORLDGEN_EXTRAS={int(args.worldgen_extras)}",
        str(REPO_ROOT),
    ]
    run(command)


def pull_image(args: argparse.Namespace) -> None:
    docker_available()
    remote_image = f"{ALIYUN_IMAGE}:{args.tag}"
    run(["docker", "pull", remote_image])
    if args.retag:
        run(["docker", "tag", remote_image, args.image])
        print(f"[PULL] tagged {remote_image} as {args.image}", flush=True)


def start(args: argparse.Namespace) -> None:
    docker_available()
    if args.build and not image_exists(args.image):
        build(args)
    if not image_exists(args.image):
        raise RuntimeError(f"Image does not exist: {args.image}. Run `python docker.py build` first.")
    if container_running(args.name):
        print(f"[INFO] Container is already running: {args.name}")
        return
    if container_exists(args.name):
        run(["docker", "rm", args.name])

    command = [
        "docker",
        "run",
        "--rm",
        "-dit",
        "--name",
        args.name,
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-e",
        "TORCH_HOME=/cache/torch",
        "-e",
        "MPLCONFIGDIR=/cache/matplotlib",
        *mount_args(args.models, args.writable_models),
    ]
    if args.display:
        command.extend(["-e", "DISPLAY", "--mount", f"type=bind,source={Path.home() / '.Xauthority'},target=/root/.Xauthority"])
        command.extend(["--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix"])
    if args.user:
        command.extend(["--user", args.user])
    command.extend(["--entrypoint", "bash", args.image, "-lc", "sleep infinity"])
    run(command)


def enter(args: argparse.Namespace) -> None:
    docker_available()
    if not container_running(args.name):
        raise RuntimeError(f"Container is not running: {args.name}. Run `python docker.py start` first.")
    run(["docker", "exec", "-it", args.name, "bash", "-lc", f"cd {CONTAINER_WORKDIR} && exec bash"])


def exec_cmd(args: argparse.Namespace) -> None:
    docker_available()
    if not container_running(args.name):
        raise RuntimeError(f"Container is not running: {args.name}. Run `python docker.py start` first.")
    command = " ".join(shlex.quote(part) for part in args.command)
    run(["docker", "exec", "-it", args.name, "bash", "-lc", command])


def download_models(args: argparse.Namespace) -> None:
    modelscope = modelscope_command(args.dry_run)
    hf_download = huggingface_download_command(args.dry_run)
    root = args.path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] target root: {root}", flush=True)
    for modelscope_id, hf_id, local_dir_name, include_patterns in MODEL_DOWNLOADS:
        local_dir = root / local_dir_name
        required_files = MODEL_REQUIRED_FILES[local_dir_name]
        exists = all((local_dir / required).is_file() for required in required_files)
        if exists and not args.force:
            print(f"[DOWNLOAD] skip existing: {local_dir}", flush=True)
            continue
        modelscope_download = [
            *modelscope,
            "download",
            "--model",
            modelscope_id,
            "--local_dir",
            str(local_dir),
        ]
        if include_patterns:
            modelscope_download.extend(["--include", *include_patterns])
        hf_fallback = [
            *hf_download,
            "download",
            hf_id,
            "--local-dir",
            str(local_dir),
        ]
        for pattern in include_patterns:
            hf_fallback.extend(["--include", pattern])
        if args.force:
            hf_fallback.append("--force-download")
        if args.dry_run:
            print("+", shlex.join(modelscope_download), flush=True)
            print("+", shlex.join(["HF_ENDPOINT=" + args.hf_endpoint, *hf_fallback]), flush=True)
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(local_dir, os.W_OK):
            raise PermissionError(f"Model directory is not writable: {local_dir}")
        try:
            run(modelscope_download)
        except subprocess.CalledProcessError:
            print(
                f"[DOWNLOAD] ModelScope failed for {modelscope_id}; falling back to Hugging Face {hf_id} "
                f"with HF_ENDPOINT={args.hf_endpoint}",
                flush=True,
            )
            env = {**os.environ, "HF_ENDPOINT": args.hf_endpoint}
            try:
                run(hf_fallback, env=env)
            except subprocess.CalledProcessError:
                print(
                    f"[DOWNLOAD] Hugging Face CLI failed for {hf_id}; downloading required files directly.",
                    flush=True,
                )
                direct_hf_download_required_files(hf_id, local_dir, required_files, args.hf_endpoint)


def docker_exec(args: argparse.Namespace, command: str) -> None:
    run(["docker", "exec", args.name, "bash", "-lc", command])


def stage(args: argparse.Namespace, label: str, command: str) -> None:
    print(f"\n[RUN] {label}", flush=True)
    recorder = getattr(args, "_profile_recorder", None)
    if recorder is None:
        docker_exec(args, f"set -euo pipefail; {command}")
        return
    record, profiled_command = recorder.wrap_stage(label, command)
    try:
        docker_exec(args, f"set -euo pipefail; {profiled_command}")
    finally:
        recorder.finish_stage(record)


def skipped_stage(args: argparse.Namespace, stage_number: int, label: str) -> None:
    print(f"\n[RUN] Skip Stage {stage_number}: {label}", flush=True)
    recorder = getattr(args, "_profile_recorder", None)
    if recorder is not None:
        recorder.record_skipped(stage_number, label)


def parse_skip_stages(spec: str) -> set[int]:
    stages: set[int] = set()
    if not spec.strip():
        return stages
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError("--skip must be a comma-separated list of stage numbers, e.g. 1,2,4.")
        stage_number = int(part)
        if stage_number < 1 or stage_number > 4:
            raise ValueError("--skip only supports worldgen stages 1 through 4.")
        stages.add(stage_number)
    return stages


def resolve_stage3_offload_mode(requested_mode: str, *, single_gpu: bool) -> str:
    if single_gpu:
        return "group-stream" if requested_mode == "auto" else requested_mode
    if requested_mode not in ("auto", "none"):
        raise ValueError(
            "--stage3-offload-mode model/sequential/block/group-stream is only supported for single-GPU Stage 3 runs."
        )
    return "none"


def vlm_stop_command(scene: str) -> str:
    return (
        f"pid_file={shlex.quote(scene + '/vlm_server.pid')}; "
        "pids=''; "
        "if [ -f \"$pid_file\" ]; then pids=\"$pids $(cat \"$pid_file\")\"; fi; "
        "if command -v pgrep >/dev/null 2>&1; then "
        "pids=\"$pids $(pgrep -f '[v]lm_server.py|[l]aunch_vlm.sh' || true)\"; "
        "fi; "
        "pids=$(printf '%s\\n' $pids | awk '/^[0-9]+$/ && !seen[$1]++'); "
        "if [ -n \"$pids\" ]; then "
        "kill $pids 2>/dev/null || true; "
        "for i in $(seq 1 30); do "
        "alive=''; "
        "for pid in $pids; do if kill -0 \"$pid\" 2>/dev/null; then alive=1; fi; done; "
        "if [ -n \"$alive\" ]; then sleep 1; else break; fi; "
        "done; "
        "for pid in $pids; do if kill -0 \"$pid\" 2>/dev/null; then kill -9 \"$pid\" 2>/dev/null || true; fi; done; "
        "echo \"VLM shim stopped: $(printf '%s' \"$pids\" | tr '\\n' ' ')\"; "
        "else "
        "echo 'No VLM shim process found to stop'; "
        "fi; "
        "rm -f \"$pid_file\"; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "echo 'GPU processes after VLM stop:'; "
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true; "
        "fi"
    )


def run_workflow(args: argparse.Namespace) -> None:
    docker_available()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.runname):
        raise ValueError("--runname may only contain letters, numbers, dot, underscore, and dash.")
    devices = [part.strip() for part in args.device.split(",") if part.strip()]
    if not devices:
        raise ValueError("--device must be a CUDA device list such as 0 or 0,1.")
    for part in devices:
        if not part.isdigit():
            raise ValueError("--device must be a comma-separated list of CUDA device ids, e.g. 0 or 0,1.")
    if args.batchsize < 1:
        raise ValueError("--batchsize must be a positive integer.")
    if args.profile_interval <= 0:
        raise ValueError("--profile-interval must be positive.")
    primary_device = devices[0]
    vlm_device = devices[1] if len(devices) > 1 else devices[0]
    all_devices = ",".join(devices)
    nproc = str(len(devices))
    stage3_single_gpu = len(devices) == 1
    stage3_launcher = f"torchrun --nproc_per_node={nproc} video_gen.py"
    stage3_fsdp = "" if stage3_single_gpu else " --fsdp"
    stage3_offload_mode = resolve_stage3_offload_mode(args.stage3_offload_mode, single_gpu=stage3_single_gpu)
    skip_stages = parse_skip_stages(args.skip)

    if not image_exists(args.image):
        raise RuntimeError(f"Image does not exist: {args.image}. Run `python docker.py build` first.")

    started_container = False
    if container_running(args.name):
        pass
    elif container_exists(args.name):
        run(["docker", "start", args.name])
    else:
        args.build = False
        args.writable_models = True
        start(args)
        started_container = True

    scene = f"{CONTAINER_WORKDIR}/examples/worldgen/{args.runname}"
    result_dir = f"{scene}/gs_results"
    prompt = args.prompt
    skip_existing_arg = " --skip_exist" if args.skip_existing else ""
    flux_panorama_lora_path = f"{CONTAINER_MODELS}/{FLUX_PANORAMA_LORA_DIR}"
    flux_panorama_lora_file = f"{flux_panorama_lora_path}/{FLUX_PANORAMA_LORA_WEIGHT}"
    print(f"[RUN] container: {args.name}", flush=True)
    print(f"[RUN] scene:     {scene}", flush=True)
    print(f"[RUN] devices:   {all_devices}", flush=True)
    print(f"[RUN] panorama:  {args.panorama_backend}", flush=True)
    print(f"[RUN] vlm:       {DEFAULT_VLM} ({DEFAULT_VLM_PATH})", flush=True)
    if skip_stages:
        print(f"[RUN] skip:      {','.join(str(stage_number) for stage_number in sorted(skip_stages))}", flush=True)
    if stage3_single_gpu:
        print(f"[RUN] stage3:   single GPU, offload mode {stage3_offload_mode}", flush=True)
    else:
        print("[RUN] stage3:   multi GPU, FSDP enabled", flush=True)

    profile_recorder = None
    if args.profile:
        profile_recorder = ProfileRecorder(
            args=args,
            scene=scene,
            devices=devices,
            stage3_offload_mode=stage3_offload_mode,
            skip_stages=skip_stages,
        )
        args._profile_recorder = profile_recorder
        print(f"[RUN] profile:  {profile_recorder.host_dir}", flush=True)

    run_status = "completed"
    run_error = None
    try:
        stage(
            args,
            "Prepare scene directory",
            shell_join("mkdir", "-p", scene),
        )

        if 1 in skip_stages:
            skipped_stage(args, 1, "panorama generation")
        elif args.panorama_backend == "hypano":
            stage(
                args,
                "Stage 1: text prompt -> condition image (FLUX.2 Klein)",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR)} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n flux2 "
                    "python -u scripts/flux2_klein_text2img.py "
                    "--model 9b "
                    "--model-path /models/FLUX.2-klein-9B "
                    f"--prompt {shlex.quote(prompt)} "
                    f"--output {shlex.quote(scene + '/condition.png')} "
                    f"--height {args.condition_height} "
                    f"--width {args.condition_width} "
                    f"--steps {args.flux_steps} "
                    f"--guidance-scale {args.flux_guidance_scale} "
                    f"--seed {args.seed} "
                    "--placement offload"
                ),
            )

            stage(
                args,
                "Stage 1: condition image + prompt -> panorama (HY-Pano)",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/panogen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2-pano "
                    "python -u pipeline_with_qwen_image.py "
                    f"--image {shlex.quote(scene + '/condition.png')} "
                    f"--prompt {shlex.quote(prompt)} "
                    f"--save {shlex.quote(scene + '/panorama.png')} "
                    "--pretrained-model-name-or-path /models/Qwen/Qwen-Image-Edit-2509 "
                    "--lora-path /models/HY-World-2.0 "
                    "--lora-subfolder HY-Pano-2.0 "
                    f"--height {args.pano_height} "
                    f"--width {args.pano_width} "
                    f"--num-inference-steps {args.pano_steps} "
                    "--load-strategy sequential-offload "
                    f"--seed {args.seed} "
                    "--reproduce"
                ),
            )
        elif args.panorama_backend == "flux-lora":
            stage(
                args,
                "Stage 1: ensure FLUX.2 9B panorama LoRA",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR)} && "
                    f"if [ ! -f {shlex.quote(flux_panorama_lora_file)} ]; then "
                    f"mkdir -p {shlex.quote(flux_panorama_lora_path)} && "
                    f"{CONDA} run --no-capture-output -n flux2 python - <<'PY'\n"
                    "from huggingface_hub import snapshot_download\n"
                    f"snapshot_download(repo_id={FLUX_PANORAMA_LORA_REPO!r}, "
                    f"local_dir={flux_panorama_lora_path!r}, "
                    f"allow_patterns=[{FLUX_PANORAMA_LORA_WEIGHT!r}])\n"
                    "PY\n"
                    "fi && "
                    f"{CONDA} run --no-capture-output -n flux2 python - <<'PY'\n"
                    "import importlib.util, subprocess, sys\n"
                    "if importlib.util.find_spec('peft') is None:\n"
                    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'peft'])\n"
                    "PY"
                ),
            )

            stage(
                args,
                "Stage 1: text prompt -> panorama (FLUX.2 Klein 9B panorama LoRA)",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR)} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n flux2 "
                    "python -u scripts/flux2_klein_text2img.py "
                    "--model 9b "
                    "--model-path /models/FLUX.2-klein-9B "
                    f"--lora-path {shlex.quote(flux_panorama_lora_path)} "
                    f"--lora-weight-name {shlex.quote(FLUX_PANORAMA_LORA_WEIGHT)} "
                    "--panorama-trigger "
                    f"--prompt {shlex.quote(prompt)} "
                    f"--output {shlex.quote(scene + '/panorama.png')} "
                    f"--height {args.flux_pano_height} "
                    f"--width {args.flux_pano_width} "
                    f"--circular-blend-width {args.flux_pano_blend_width} "
                    f"--steps {args.flux_pano_steps} "
                    f"--guidance-scale {args.flux_guidance_scale} "
                    f"--seed {args.seed} "
                    "--device cuda:0 "
                    "--placement offload"
                ),
            )
        else:
            raise ValueError(f"Unsupported panorama backend: {args.panorama_backend!r}")

        if 2 not in skip_stages:
            stage(
                args,
                "Stage 2: start VLM shim for WorldNav",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR)} && "
                    f"mkdir -p {shlex.quote(scene)} && "
                    f"nohup env CUDA_VISIBLE_DEVICES={shlex.quote(vlm_device)} PORT={args.vlm_port} "
                    f"scripts/launch_vlm.sh > {shlex.quote(scene + '/vlm_server.log')} 2>&1 & "
                    f"echo $! > {shlex.quote(scene + '/vlm_server.pid')}; "
                    f"{CONDA} run --no-capture-output -n hyworld2 python - <<'PY'\n"
                    "import pathlib, sys, time, urllib.request\n"
                    f"log = pathlib.Path({scene + '/vlm_server.log'!r})\n"
                    f"url = 'http://127.0.0.1:{args.vlm_port}/health'\n"
                    "for attempt in range(120):\n"
                    "    try:\n"
                    "        urllib.request.urlopen(url, timeout=2).read()\n"
                    "        print('VLM shim healthy', flush=True)\n"
                    "        break\n"
                    "    except Exception:\n"
                    "        if attempt % 6 == 0:\n"
                    "            print(f'waiting for VLM shim... {attempt * 5}s', flush=True)\n"
                    "        time.sleep(5)\n"
                    "else:\n"
                    "    print('VLM shim failed to become healthy. Last log lines:', flush=True)\n"
                    "    if log.exists():\n"
                    "        print('\\n'.join(log.read_text(errors='replace').splitlines()[-80:]), flush=True)\n"
                    "    sys.exit(1)\n"
                    "PY"
                ),
            )

            stage(
                args,
                "Stage 2: trajectory planning",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/worldgen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                    "python -u traj_generate.py "
                    f"--target_path {shlex.quote(scene)} "
                    f"--llm_addr localhost --llm_port {args.vlm_port} --llm_name {shlex.quote(DEFAULT_VLM)} "
                    f"--apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm{skip_existing_arg}"
                ),
            )

            stage(
                args,
                "Stage 2: trajectory rendering and VLM captions",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/worldgen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                    "torchrun --nproc_per_node=1 traj_render.py "
                    f"--target_path {shlex.quote(scene)} "
                    f"--llm_addr localhost --llm_port {args.vlm_port} --llm_name {shlex.quote(DEFAULT_VLM)}"
                ),
            )

            stage(
                args,
                "Stage 2: stop VLM shim before Stage 3",
                vlm_stop_command(scene),
            )
        else:
            skipped_stage(args, 2, "trajectory planning, rendering, and VLM captions")
            if 3 not in skip_stages:
                stage(args, "Ensure VLM shim is stopped before Stage 3", vlm_stop_command(scene))

        if 3 in skip_stages:
            skipped_stage(args, 3, "WorldStereo expansion and WorldMirror generation bank")
        else:
            stage(
                args,
                "Stage 3: WorldStereo expansion and WorldMirror generation bank",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/worldgen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(all_devices)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                    f"{stage3_launcher} "
                    f"--target_path {shlex.quote(scene)}{stage3_fsdp} --local_files_only "
                    f"--offload-mode {stage3_offload_mode}{skip_existing_arg}"
                ),
            )

        if 4 in skip_stages:
            skipped_stage(args, 4, "prepare 3DGS training data, train and export 3DGS")
        else:
            stage(
                args,
                "Stage 4: prepare 3DGS training data",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/worldgen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(all_devices)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                    f"torchrun --nproc_per_node={nproc} gen_gs_data.py "
                    f"--root_path {shlex.quote(scene)} --save_normal --split_sky"
                ),
            )

            stage(
                args,
                "Stage 4: train and export 3DGS",
                (
                    f"cd {shlex.quote(CONTAINER_WORKDIR + '/hyworld2/worldgen')} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(primary_device)} "
                    "/opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 "
                    "python -u -m world_gs_trainer default "
                    f"--data-dir {shlex.quote(scene + '/gs_data')} "
                    f"--result-dir {shlex.quote(result_dir)} "
                    f"--max-steps {args.gs_steps} --save-steps {args.gs_steps} "
                    f"--eval-steps {args.gs_steps} --ply-steps {args.gs_steps} "
                    f"--batch-size {args.batchsize} "
                    "--save-ply --convert-to-spz --disable-video --disable-viewer "
                    "--use-scale-regularization --antialiased "
                    "--depth-loss --normal-loss --sky-depth-from-pcd "
                    "--use-mask-gaussian --mask-export-stochastic "
                    "--no-mask-export-anchor-protection --use-anchor-protection --export-mesh "
                    "--ssim-lambda 0 "
                    "--strategy.refine-start-iter 150 "
                    "--strategy.refine-stop-iter 2000 "
                    "--strategy.refine-every 100 "
                    "--strategy.refine-scale2d-stop-iter 2000 "
                    "--strategy.reset-every 99990 "
                    "--strategy.grow-grad2d 0.0001 "
                    "--strategy.prune-scale3d 0.1"
                ),
            )

        print(f"\n[RUN] complete: {scene}", flush=True)
        print(f"[RUN] 3DGS result: {result_dir}", flush=True)
    except BaseException as exc:
        run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if profile_recorder is not None:
            profile_recorder.write(status=run_status, error=run_error)
        if started_container:
            print(f"\n[RUN] stopping temporary container: {args.name}", flush=True)
            stop(args)


def stop(args: argparse.Namespace) -> None:
    docker_available()
    if container_running(args.name) or container_exists(args.name):
        run(["docker", "rm", "-f", args.name])
    else:
        print(f"[INFO] Container does not exist: {args.name}")


def status(args: argparse.Namespace) -> None:
    docker_available()
    print(f"image:     {args.image} ({'present' if image_exists(args.image) else 'missing'})")
    if container_running(args.name):
        container_status = "running"
    elif container_exists(args.name):
        container_status = "stopped"
    else:
        container_status = "missing"
    print(f"container: {args.name} ({container_status})")


def verify(args: argparse.Namespace) -> None:
    docker_available()
    if not container_running(args.name):
        raise RuntimeError(f"Container is not running: {args.name}. Run `python docker.py start` first.")

    checks = [
        (
            "hyworld2 env",
            "cd /workspace/hyworld2 && "
            "/opt/miniconda3/bin/conda run -n hyworld2 python -c "
            "\"import torch, diffusers, transformers, recast, gsplat; "
            "import hyworld2.worldrecon.pipeline; "
            "print('hyworld2 ok', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), diffusers.__version__, transformers.__version__)\"",
        ),
        (
            "objgen runtime",
            "cd /workspace/hyworld2 && "
            "/opt/miniconda3/bin/conda run -n hyworld2 python -c "
            "\"import sam3d_objects; from hyworld2.objgen.inference import Inference; "
            "print('objgen ok', sam3d_objects.__file__, Inference.__name__)\"",
        ),
        (
            "worldgen runtime defaults",
            "/opt/miniconda3/bin/conda run -n hyworld2 python -c "
            "\"import os; "
            "expected={'HF_HOME':'/models/.cache/huggingface',"
            "'HUGGINGFACE_HUB_CACHE':'/models/.cache/huggingface/hub',"
            "'HF_ENDPOINT':'https://hf-mirror.com',"
            "'SAM3_REPO_ID':'/models/sam3',"
            "'WORLDSTEREO_REPO':'/models/WorldStereo',"
            "'WORLDSTEREO_BASE_MODEL':'/models/Wan2.1-I2V-14B-480P-Diffusers',"
            "'WORLDMIRROR_MODEL':'/models/HY-World-2.0',"
            "'MOGE_MODEL':'/models/moge-2-vitl-normal',"
            "'ZIM_MODEL':'/models/zim-anything-vitl',"
            "'GROUNDING_DINO_MODEL':'/models/grounding-dino-tiny',"
            "'CAMERA_SELECTOR_MODEL':'/models/dinov2-base',"
            "'WS_TEXT_DTYPE':'bf16',"
            "'WS_AUX_OFFLOAD':'1',"
            "'WORLDMIRROR_NPROC_PER_NODE':'1',"
            "'WORLDMIRROR_TARGET_SIZE':'512',"
            "'WORLDMIRROR_CUDA_VISIBLE_DEVICES':'0'}; "
            "bad={k:(os.environ.get(k),v) for k,v in expected.items() if os.environ.get(k)!=v}; "
            "assert not bad, bad; "
            "print('worldgen defaults ok', ' '.join(f'{k}={v}' for k,v in expected.items()))\"",
        ),
        (
            "PyTorch3D CUDA rasterizer",
            "cd /workspace/hyworld2 && "
            "/opt/miniconda3/bin/conda run -n hyworld2 python -c "
            "\"import torch; "
            "from pytorch3d.renderer import PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer; "
            "from pytorch3d.structures import Pointclouds; "
            "points=torch.tensor([[0.0,0.0,2.0],[0.1,0.0,2.0]], device='cuda'); "
            "cloud=Pointclouds(points=[points]); "
            "cameras=PerspectiveCameras(device='cuda'); "
            "settings=PointsRasterizationSettings(image_size=4, radius=0.1, points_per_pixel=2); "
            "fragments=PointsRasterizer(cameras=cameras, raster_settings=settings)(cloud); "
            "torch.cuda.synchronize(); "
            "print('pytorch3d cuda rasterizer ok', tuple(fragments.idx.shape))\"",
        ),
        (
            "hyworld2-pano env",
            "cd /workspace/hyworld2/hyworld2/panogen && "
            "/opt/miniconda3/bin/conda run -n hyworld2-pano python -c "
            "\"import torch, pipeline, pipeline_with_qwen_image; "
            "print('hypano ok', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())\"",
        ),
        (
            "model mount",
            "test -d /models && echo 'models ok:' && find /models -maxdepth 3 \\( -type d -o -type f \\) | sort | head -40",
        ),
    ]
    for label, command in checks:
        print(f"\n[VERIFY] {label}")
        run(["docker", "exec", args.name, "bash", "-lc", command])


def add_action_parsers(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false", default=True)
    build_parser.add_argument("--no-worldgen-extras", dest="worldgen_extras", action="store_false", default=True)
    build_parser.set_defaults(func=build)

    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull the published HY-World image from the Aliyun registry.",
    )
    pull_parser.add_argument(
        "--tag",
        default=image_tag(DEFAULT_IMAGE),
        help=f"Registry tag to pull from {ALIYUN_IMAGE}.",
    )
    pull_parser.add_argument(
        "--no-retag",
        dest="retag",
        action="store_false",
        default=True,
        help="Do not retag the pulled image as --image.",
    )
    pull_parser.set_defaults(func=pull_image)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--build", action="store_true", help="Build the image first if it is missing.")
    start_parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    start_parser.add_argument("--writable-models", action="store_true")
    start_parser.add_argument("--display", action="store_true", help="Forward X11 display mounts.")
    start_parser.add_argument("--user", default=None, help="Optional Docker --user value, e.g. $(id -u):1000.")
    start_parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false", default=True)
    start_parser.add_argument("--no-worldgen-extras", dest="worldgen_extras", action="store_false", default=True)
    start_parser.set_defaults(func=start)

    subparsers.add_parser("enter").set_defaults(func=enter)

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("command", nargs=argparse.REMAINDER)
    exec_parser.set_defaults(func=exec_cmd)

    subparsers.add_parser("verify").set_defaults(func=verify)
    subparsers.add_parser("status").set_defaults(func=status)
    subparsers.add_parser("stop").set_defaults(func=stop)

    download_parser = subparsers.add_parser(
        "download",
        help="Download all models required by the prompt-to-3DGS workflow via ModelScope.",
    )
    download_parser.add_argument("--path", type=Path, required=True, help="Target model root, e.g. ./models.")
    download_parser.add_argument("--dry-run", action="store_true", help="Print ModelScope commands without downloading.")
    download_parser.add_argument("--force", action="store_true", help="Download even when the target directory is non-empty.")
    download_parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF_ENDPOINT used by the Hugging Face fallback when ModelScope has no matching repo.",
    )
    download_parser.set_defaults(func=download_models)

    run_parser = subparsers.add_parser(
        "run",
        help="Run prompt -> condition image -> panorama -> 3DGS inside the container.",
    )
    run_parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    run_parser.add_argument("--runname", required=True, help="Scene directory name under examples/worldgen.")
    run_parser.add_argument("--device", default="0", help="CUDA devices to use, e.g. 0 or 0,1.")
    run_parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    run_parser.add_argument("--writable-models", action="store_true")
    run_parser.add_argument("--display", action="store_true", help="Forward X11 display mounts if the container is started.")
    run_parser.add_argument("--user", default=None, help="Optional Docker --user value if the container is started.")
    run_parser.add_argument(
        "--panorama-backend",
        choices=("hypano", "flux-lora"),
        default="hypano",
        help="Panorama generation path: FLUX condition image + HY-Pano, or direct FLUX.2 9B panorama LoRA.",
    )
    run_parser.add_argument("--flux-steps", type=int, default=4)
    run_parser.add_argument("--flux-guidance-scale", type=float, default=1.0)
    run_parser.add_argument("--condition-height", type=int, default=1024)
    run_parser.add_argument("--condition-width", type=int, default=1024)
    run_parser.add_argument("--flux-pano-steps", type=int, default=4)
    run_parser.add_argument("--flux-pano-height", type=int, default=960)
    run_parser.add_argument("--flux-pano-width", type=int, default=1952)
    run_parser.add_argument("--flux-pano-blend-width", type=int, default=32)
    run_parser.add_argument("--pano-steps", type=int, default=40)
    run_parser.add_argument("--pano-height", type=int, default=960)
    run_parser.add_argument("--pano-width", type=int, default=1952)
    run_parser.add_argument("--gs-steps", type=int, default=4000)
    run_parser.add_argument("--batchsize", type=int, default=4, help="Per-GPU 3DGS training batch size.")
    # Differs from upstream 8-GPU example max_steps=1500; local runs usually trade more steps for fewer GPUs.
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--vlm-port", type=int, default=8000)
    # Local wrapper only: upstream starts the OpenAI-compatible VLM server separately.
    run_parser.add_argument("--skip-existing", action="store_true", help="Pass --skip_exist to resumable worldgen stages.")
    # Local resume helper: forwards --skip_exist to stages that support it.
    run_parser.add_argument("--skip", default="", help="Comma-separated worldgen stages to skip, e.g. 1,2,4.")
    run_parser.add_argument("--profile", action="store_true", help="Record per-stage runtime and GPU usage under the scene profiles directory.")
    run_parser.add_argument("--profile-interval", type=float, default=1.0, help="GPU profiling sample interval in seconds.")
    run_parser.add_argument(
        "--stage3-offload-mode",
        choices=("auto", "none", "model", "sequential", "block", "group-stream"),
        default="auto",
        help="WorldStereo Stage 3 offload mode. auto uses group-stream on one GPU and none with multi-GPU FSDP.",
    )
    run_parser.set_defaults(func=run_workflow)


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.image is None:
        args.image = DEFAULT_IMAGE
    if args.name is None:
        args.name = DEFAULT_CONTAINER
    if args.dockerfile is None:
        args.dockerfile = DEFAULT_DOCKERFILE
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the HY-World container."
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dockerfile", type=Path, default=None)

    add_action_parsers(parser)
    return apply_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
