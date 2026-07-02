from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from text2scene import load_scene_config


def get_target_size(scene_config_path: Path, index: int) -> tuple[str, np.ndarray]:
    config = load_scene_config(scene_config_path)
    obstacles = config["obstacles"]
    if index < 0 or index >= len(obstacles):
        raise IndexError(f"index out of range: {index}, obstacles={len(obstacles)}")
    obstacle = obstacles[index]
    real_size = obstacle["real_size"]
    return (
        obstacle["name"],
        np.array(
            [
                real_size["length_m"],
                real_size["width_m"],
                real_size["height_m"],
            ],
            dtype=np.float64,
        ),
    )


def get_vertices_array(ply: PlyData) -> tuple[np.ndarray, np.ndarray]:
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    vertex = ply["vertex"].data
    for key in ("x", "y", "z"):
        if key not in vertex.dtype.names:
            raise ValueError(f"PLY vertex has no {key} field")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )
    return vertex, xyz


def compute_bbox(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    return min_xyz, max_xyz, max_xyz - min_xyz


def normalize_to_ground_center(xyz: np.ndarray) -> np.ndarray:
    min_xyz, max_xyz, _ = compute_bbox(xyz)
    normalized = xyz.copy()
    normalized[:, 0] -= (min_xyz[0] + max_xyz[0]) / 2.0
    normalized[:, 1] -= (min_xyz[1] + max_xyz[1]) / 2.0
    normalized[:, 2] -= min_xyz[2]
    return normalized


def scale_by_volume_ratio(
    xyz: np.ndarray,
    target_size: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_min, current_max, current_size = compute_bbox(xyz)
    current_size = np.maximum(current_size, 1e-8)
    current_volume = float(np.prod(current_size))
    target_volume = float(np.prod(target_size))
    if current_volume <= 0:
        raise ValueError(f"invalid current PLY bbox volume: {current_volume}")
    scale = (target_volume / current_volume) ** (1.0 / 3.0)
    scaled = xyz * scale
    final_min, final_max, final_size = compute_bbox(scaled)
    return scaled, {
        "scale_mode": "volume",
        "target_size": target_size.tolist(),
        "source_bbox_min": current_min.tolist(),
        "source_bbox_max": current_max.tolist(),
        "source_bbox_size": current_size.tolist(),
        "source_volume": current_volume,
        "target_volume": target_volume,
        "scale": scale,
        "final_bbox_min": final_min.tolist(),
        "final_bbox_max": final_max.tolist(),
        "final_bbox_size": final_size.tolist(),
    }


def write_ply_like_original(input_ply: PlyData, output_path: Path, new_xyz: np.ndarray) -> None:
    vertex_data = input_ply["vertex"].data.copy()
    vertex_data["x"] = new_xyz[:, 0].astype(vertex_data["x"].dtype)
    vertex_data["y"] = new_xyz[:, 1].astype(vertex_data["y"].dtype)
    vertex_data["z"] = new_xyz[:, 2].astype(vertex_data["z"].dtype)

    elements = []
    for element in input_ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex_data, "vertex"))
        else:
            elements.append(element)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        elements,
        text=input_ply.text,
        byte_order=input_ply.byte_order,
        comments=input_ply.comments,
        obj_info=input_ply.obj_info,
    ).write(str(output_path))


def calibrate_ply(
    input_ply_path: Path,
    output_ply_path: Path,
    scene_config_path: Path,
    index: int,
) -> dict[str, Any]:
    name, target_size = get_target_size(scene_config_path, index)
    ply = PlyData.read(str(input_ply_path))
    _, xyz = get_vertices_array(ply)
    source_min, source_max, source_size = compute_bbox(xyz)
    normalized_xyz = normalize_to_ground_center(xyz)
    final_xyz, stats = scale_by_volume_ratio(normalized_xyz, target_size)
    write_ply_like_original(ply, output_ply_path, final_xyz)
    stats["obstacle_name"] = name
    stats["original_bbox_min"] = source_min.tolist()
    stats["original_bbox_max"] = source_max.tolist()
    stats["original_bbox_size"] = source_size.tolist()
    stats["output"] = str(output_ply_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground, center, and scale a SAM3D PLY from scene_config.json.")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = calibrate_ply(
        Path(args.input),
        Path(args.output),
        Path(args.scene_config),
        args.index,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
