"""Online receding-horizon controller loop for Go2 high-level velocity control."""

from __future__ import annotations

import logging
import time

import torch

from deploy.go2.go2_state_reader import Go2StateReader
from deploy.go2.go2_velocity_commander import Go2VelocityCommander
from deploy.go2.safety_filter import VelocitySafetyFilter
from robotics.data.prompt_builder import RoboticsPromptBuilder, resolve_mask_token_text, tokenize_prompt_segments

LOGGER = logging.getLogger(__name__)


class Go2ControllerLoop:
    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        reader: Go2StateReader,
        commander: Go2VelocityCommander,
        safety_filter: VelocitySafetyFilter,
        instruction: str,
        execute_steps: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.reader = reader
        self.commander = commander
        self.safety_filter = safety_filter
        self.instruction = instruction
        self.execute_steps = execute_steps
        self.prompt_builder = RoboticsPromptBuilder(
            model.velocity_config,
            mask_token_text=resolve_mask_token_text(tokenizer, fallback_token_id=model.velocity_config.mask_token_id),
        )

    def _attach_model_debug(self, debug_payload: dict[str, object]) -> None:
        self.commander.adapter.set_model_debug(debug_payload)

    @staticmethod
    def _maybe_tensor_to_list(outputs: dict[str, object], key: str):
        value = outputs.get(key)
        if value is None:
            return None
        return value[0].detach().cpu().tolist()

    def run_once(self) -> torch.Tensor:
        loop_start = time.perf_counter()
        observation = self.reader.read()
        image_missing = observation.image is None
        previous_action = {
            "vx": float(observation.state.get("vx_prev", 0.0)),
            "vy": float(observation.state.get("vy_prev", 0.0)),
            "wz": float(observation.state.get("wz_prev", observation.state.get("yaw_speed", observation.state.get("wz", 0.0)))),
        }
        prompt_result = self.prompt_builder.build(
            self.instruction,
            observation.state,
            previous_action=previous_action,
            include_image=not image_missing,
        )
        tokenized = tokenize_prompt_segments(prompt_result, self.tokenizer)
        input_ids = tokenized.input_ids.unsqueeze(0)
        action_positions = tokenized.action_positions.unsqueeze(0)
        images = None
        image_sizes = None
        modalities = None
        if not image_missing:
            pixel_values = self.image_processor.preprocess(observation.image, return_tensors="pt")["pixel_values"][0]
            images = [pixel_values.to(next(self.model.parameters()).device)]
            image_sizes = [observation.image.size]
            modalities = ["image"]
        predict_start = time.perf_counter()
        outputs = self.model.predict_actions(
            input_ids=input_ids.to(next(self.model.parameters()).device),
            attention_mask=torch.ones_like(input_ids).to(next(self.model.parameters()).device),
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            action_positions=action_positions.to(next(self.model.parameters()).device),
            previous_actions=torch.tensor(
                [
                    [
                        float(observation.state.get("vx_prev", 0.0)),
                        float(observation.state.get("vy_prev", 0.0)),
                        float(observation.state.get("wz_prev", observation.state.get("yaw_speed", observation.state.get("wz", 0.0)))),
                    ]
                ],
                dtype=torch.float32,
                device=next(self.model.parameters()).device,
            ),
            states=[observation.state],
            execute_steps=self.execute_steps,
        )
        predict_ms = (time.perf_counter() - predict_start) * 1000.0
        execute_actions = outputs["execute_actions"][0].detach().cpu()
        safe_commands = []
        for step_idx, command in enumerate(execute_actions):
            if step_idx > 0:
                observation = self.reader.read()
                image_missing = observation.image is None
            safety_start = time.perf_counter()
            safe = self.safety_filter.apply(command, observation.state, image_missing=image_missing, image_timestamp=observation.timestamp)
            safety_ms = (time.perf_counter() - safety_start) * 1000.0
            total_loop_ms = (time.perf_counter() - loop_start) * 1000.0
            self._attach_model_debug(
                {
                    "instruction": self.instruction,
                    "previous_action": previous_action,
                    "image_missing": image_missing,
                    "execute_steps": int(self.execute_steps),
                    "execute_step_index": int(step_idx),
                    "predict_ms": float(predict_ms),
                    "safety_ms": float(safety_ms),
                    "control_loop_ms": float(total_loop_ms),
                    "target_actions": self._maybe_tensor_to_list(outputs, "target_actions"),
                    "anchor_actions": self._maybe_tensor_to_list(outputs, "anchor_actions"),
                    "residual_actions": self._maybe_tensor_to_list(outputs, "residual_actions"),
                    "continuous_actions": self._maybe_tensor_to_list(outputs, "continuous_actions"),
                    "execute_actions": self._maybe_tensor_to_list(outputs, "execute_actions"),
                    "step_confidence": self._maybe_tensor_to_list(outputs, "step_confidence"),
                    "selected_execute_action": command.tolist(),
                    "safe_command": safe.command.tolist(),
                    "safety_mode": safe.mode,
                    "safety_reason": safe.reason,
                }
            )
            self.commander.send(float(safe.command[0]), float(safe.command[1]), float(safe.command[2]))
            safe_commands.append(safe.command)
            LOGGER.info("sent command=%s mode=%s reason=%s", safe.command.tolist(), safe.mode, safe.reason)
        if len(safe_commands) == 1:
            return safe_commands[0]
        return torch.stack(safe_commands, dim=0)

    def run(self, iterations: int = 10) -> None:
        sleep_s = 1.0 / self.model.velocity_config.control_hz
        for _ in range(iterations):
            self.run_once()
            time.sleep(sleep_s)
