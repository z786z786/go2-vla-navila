"""Fail-closed specification for the independent N1 semantic preview camera."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POLICY_RGB_PRIM = "{ENV_REGEX_NS}/Robot/base/rgbd_camera"
POLICY_RGB_OFFSET = (0.1, 0.0, 0.5, -0.5, 0.5, -0.5, 0.5)


@dataclass(frozen=True)
class PreviewCameraSpec:
    scene_id: str
    semantic_mesh_path: str
    prim_path: str = POLICY_RGB_PRIM
    width: int = 512
    height: int = 512
    data_types: tuple[str, ...] = ("semantic_segmentation",)
    policy_feature_names: tuple[str, ...] = ()
    preview_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preview_camera_spec(scene_id: str, semantic_mesh_path: Path) -> PreviewCameraSpec:
    """Bind preview pose/size to policy RGB while keeping outputs isolated."""
    return PreviewCameraSpec(scene_id=str(scene_id), semantic_mesh_path=str(semantic_mesh_path.resolve()))


def validate_preview_camera_spec(spec: PreviewCameraSpec) -> list[str]:
    errors: list[str] = []
    if not spec.preview_only or spec.policy_feature_names:
        errors.append("semantic preview must not expose any policy feature")
    if spec.prim_path != POLICY_RGB_PRIM or (spec.width, spec.height) != (512, 512):
        errors.append("semantic preview must match the policy RGB pose anchor and 512x512 image size")
    if spec.data_types != ("semantic_segmentation",):
        errors.append("preview camera may output semantic_segmentation only")
    path = Path(spec.semantic_mesh_path)
    if not path.is_file():
        errors.append(f"semantic mesh is unavailable: {path}")
    return errors


def isaac_camera_construction_contract(spec: PreviewCameraSpec) -> dict[str, Any]:
    """Declarative hand-off for N2; intentionally does not import Isaac Sim."""
    return {
        "sensor_cfg": "RayCasterCameraCfg",
        "sensor_class": "omni.isaac.matterport.domains.MatterportRayCasterCamera",
        "prim_path": spec.prim_path,
        "offset_wxyz": [POLICY_RGB_OFFSET[3], POLICY_RGB_OFFSET[4], POLICY_RGB_OFFSET[5], POLICY_RGB_OFFSET[6]],
        "offset_xyz": list(POLICY_RGB_OFFSET[:3]),
        "convention": "ros",
        "width": spec.width,
        "height": spec.height,
        "data_types": list(spec.data_types),
        "mesh_prim_paths": [spec.semantic_mesh_path],
        "policy_feature_names": [],
        "preview_only": True,
    }
