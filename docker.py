#!/usr/bin/env python3
"""Small Docker interface for the HY-World container images."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = "hyworld2-isaaclab:3.0.0-beta2"
DEFAULT_CONTAINER = "hyworld2-isaaclab"
DEFAULT_DOCKERFILE = Path("Dockerfile.isaac")
DEFAULT_MODELS = Path("/data/hyworld/models")
CONTAINER_WORKDIR = "/workspace/hyworld2"
CONTAINER_MODELS = "/models"
DEFAULT_VARIANT = "isaac"
COMMANDS = ("build", "start", "enter", "exec", "verify", "status", "stop")
VARIANTS = {
    "isaac": {
        "image": DEFAULT_IMAGE,
        "name": DEFAULT_CONTAINER,
        "dockerfile": DEFAULT_DOCKERFILE,
    },
    "base": {
        "image": "hyworld2-base:3.0.0-beta2",
        "name": "hyworld2-base",
        "dockerfile": Path("Dockerfile.base"),
    },
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(command, check=check)


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()


def docker_available() -> None:
    if not capture(["docker", "version", "--format", "{{.Server.Version}}"]):
        raise RuntimeError("Docker is not available or the current user cannot access the Docker daemon.")


def image_exists(image: str) -> bool:
    return bool(capture(["docker", "image", "inspect", image, "--format", "{{.Id}}"]))


def container_running(name: str) -> bool:
    names = capture(["docker", "ps", "--format", "{{.Names}}"]).splitlines()
    return name in names


def container_exists(name: str) -> bool:
    names = capture(["docker", "ps", "-a", "--format", "{{.Names}}"]).splitlines()
    return name in names


def using_isaac(args: argparse.Namespace) -> bool:
    return args.variant == "isaac"


def cache_root(include_isaac: bool) -> Path:
    root = Path(os.environ.get("HYWORLD_DOCKER_CACHE", "~/.cache/hyworld2-docker")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o777)
    subdirs = [
        "hf",
        "torch",
        "matplotlib",
    ]
    if include_isaac:
        subdirs.extend(
            [
                "isaac-sim/cache/kit",
                "isaac-sim/cache/ov",
                "isaac-sim/cache/pip",
                "isaac-sim/cache/glcache",
                "isaac-sim/cache/computecache",
                "isaac-sim/logs",
                "isaac-sim/data",
                "isaac-sim/documents",
            ]
        )
    for subdir in subdirs:
        path = root.joinpath(subdir)
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
    return root


def mount_args(models: Path, writable_models: bool, include_isaac: bool) -> list[str]:
    root = cache_root(include_isaac)
    mounts = [
        (REPO_ROOT, CONTAINER_WORKDIR, False),
        (models.expanduser(), CONTAINER_MODELS, not writable_models),
        (root / "hf", "/cache/huggingface", False),
        (root / "torch", "/cache/torch", False),
        (root / "matplotlib", "/cache/matplotlib", False),
    ]
    if include_isaac:
        mounts.extend(
            [
                (root / "isaac-sim/cache/kit", "/isaac-sim/kit/cache", False),
                (root / "isaac-sim/cache/ov", "/root/.cache/ov", False),
                (root / "isaac-sim/cache/pip", "/root/.cache/pip", False),
                (root / "isaac-sim/cache/glcache", "/root/.cache/nvidia/GLCache", False),
                (root / "isaac-sim/cache/computecache", "/root/.nv/ComputeCache", False),
                (root / "isaac-sim/logs", "/root/.nvidia-omniverse/logs", False),
                (root / "isaac-sim/data", "/root/.local/share/ov/data", False),
                (root / "isaac-sim/documents", "/root/Documents", False),
            ]
        )
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


def start(args: argparse.Namespace) -> None:
    docker_available()
    include_isaac = using_isaac(args)
    if args.build and not image_exists(args.image):
        build(args)
    if not image_exists(args.image):
        raise RuntimeError(f"Image does not exist: {args.image}. Run `python docker.py {args.variant} build` first.")
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
        *mount_args(args.models, args.writable_models, include_isaac),
    ]
    if include_isaac:
        command.extend(["-e", "ACCEPT_EULA=Y", "-e", "PRIVACY_CONSENT=Y"])
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


def stop(args: argparse.Namespace) -> None:
    docker_available()
    if container_running(args.name) or container_exists(args.name):
        run(["docker", "rm", "-f", args.name])
    else:
        print(f"[INFO] Container does not exist: {args.name}")


def status(args: argparse.Namespace) -> None:
    docker_available()
    print(f"image:     {args.image} ({'present' if image_exists(args.image) else 'missing'})")
    print(f"container: {args.name} ({'running' if container_running(args.name) else 'stopped'})")


def verify(args: argparse.Namespace) -> None:
    docker_available()
    if not container_running(args.name):
        raise RuntimeError(f"Container is not running: {args.name}. Run `python docker.py start` first.")

    checks = [
        (
            "Isaac Lab Python (if present)",
            "set -e; "
            "if [ -d /workspace/IsaacLab ] || [ -d /workspace/isaaclab ]; then "
            "if [ -d /workspace/IsaacLab ]; then cd /workspace/IsaacLab; else cd /workspace/isaaclab; fi; "
            "./isaaclab.sh -p -c \"import sys; import isaaclab; print('isaaclab ok', sys.executable)\"; "
            "else echo 'isaaclab skipped: not present in this image'; fi",
        ),
        (
            "hyworld2 env",
            "cd /workspace/hyworld2 && "
            "/opt/miniconda3/bin/conda run -n hyworld2 python -c "
            "\"import torch, diffusers, transformers, recast, gsplat; "
            "import hyworld2.worldrecon.pipeline; "
            "print('hyworld2 ok', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), diffusers.__version__, transformers.__version__)\"",
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
            "'WORLDMIRROR_MODEL':'/models/HY-World-2.0',"
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


def apply_variant_defaults(args: argparse.Namespace) -> argparse.Namespace:
    variant = args.variant
    config = VARIANTS[variant]
    if args.image is None:
        args.image = config["image"]
    if args.name is None:
        args.name = config["name"]
    if args.dockerfile is None:
        args.dockerfile = config["dockerfile"]
    return args


def normalize_legacy_args(argv: list[str]) -> list[str]:
    if not argv or argv[0] in ("-h", "--help"):
        return argv
    value_options = {"--image", "--name", "--dockerfile"}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in value_options:
            index += 2
            continue
        if any(item.startswith(f"{option}=") for option in value_options):
            index += 1
            continue
        if item.startswith("-"):
            index += 1
            continue
        if item in COMMANDS:
            return [*argv[:index], DEFAULT_VARIANT, *argv[index:]]
        return argv
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and run HY-World containers. Use `python docker.py base build` "
            "or `python docker.py isaac build` to choose the image variant."
        )
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dockerfile", type=Path, default=None)

    variant_parsers = parser.add_subparsers(dest="variant", required=True)
    for variant in VARIANTS:
        variant_parser = variant_parsers.add_parser(variant, help=f"Use the {variant} container image.")
        add_action_parsers(variant_parser)
    return apply_variant_defaults(parser.parse_args(normalize_legacy_args(sys.argv[1:])))


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
