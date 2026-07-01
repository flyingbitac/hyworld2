from pathlib import Path


def resolve_local_model_path(path_or_repo: str) -> str:
    path = Path(path_or_repo)
    if path.is_absolute() and not path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {path}")
    return str(path)


def resolve_moge_checkpoint(path_or_repo: str) -> str:
    path = Path(path_or_repo)
    if path.is_dir():
        checkpoint = path / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MoGe checkpoint not found: {checkpoint}")
        return str(checkpoint)
    if path.is_absolute() and not path.is_file():
        raise FileNotFoundError(f"MoGe checkpoint path does not exist: {path}")
    return str(path)
