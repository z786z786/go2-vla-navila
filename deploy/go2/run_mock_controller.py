"""Run a mock Go2 controller loop against the velocity-control model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llava.model.builder import load_pretrained_model

from deploy.go2.adapters.mock_adapter import MockRobotAdapter
from deploy.go2.controller_loop import Go2ControllerLoop
from deploy.go2.go2_state_reader import Go2StateReader
from deploy.go2.go2_velocity_commander import Go2VelocityCommander
from deploy.go2.mock_robot import MockRobot
from deploy.go2.safety_filter import VelocitySafetyFilter
from robotics.config import VelocityControlConfig
from robotics.modeling.llada_vla_velocity import LladaVLAForVelocityControl


def load_model(model_path: str, device: str):
    tokenizer, backbone, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llada-vla-go2",
        device_map="cpu",
        multimodal=True,
        torch_dtype="float16" if device.startswith("cuda") else "bfloat16",
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token
    velocity_cfg = VelocityControlConfig(mask_token_id=tokenizer.mask_token_id or 126336)
    model = LladaVLAForVelocityControl(backbone=backbone, velocity_config=velocity_cfg)
    head_path = Path(model_path) / "robotics_velocity_head.bin"
    if head_path.exists():
        model.load_state_dict(torch.load(head_path, map_location="cpu"), strict=False)
    model.to(device)
    model.eval()
    return tokenizer, image_processor, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock online controller loop")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer, image_processor, model = load_model(args.model_path, args.device)
    robot = MockRobot()
    adapter = MockRobotAdapter(robot)
    loop = Go2ControllerLoop(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        reader=Go2StateReader(adapter),
        commander=Go2VelocityCommander(adapter),
        safety_filter=VelocitySafetyFilter(model.velocity_config),
        instruction=args.instruction,
    )
    loop.run(iterations=args.iterations)
    print(robot.command_history)


if __name__ == "__main__":
    main()
