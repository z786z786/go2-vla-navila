#!/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python
"""Offline P4-T6 throughput probe.

The probe deliberately produces an honest ``MISSING`` record when CUDA or the
NVIDIA driver is unavailable.  On a live GPU it runs the mandated waiter,
re-checks admission while holding the project lease, performs >=10 warmup
steps, and records both torch and nvidia-smi memory measurements.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path("/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python")
INDEX = Path("/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl")
FRAMES = Path("/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train")
MODEL = Path("/mnt/wxh/go2_short_vln/cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467")
RUN_ID = "p4_t6_throughput"
TOTAL_SAMPLES = 353_894
TOTAL_FRAMES = 601_125
FEATURE_BYTES = 64 * 960 * 2
WARMUP_STEPS = 10
TIMED_STEPS = 10
_ACTIVE_LEASE: Any | None = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smi_sample() -> dict[str, Any] | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10
        )
        vals = [x.strip() for x in p.stdout.strip().split(",")]
        if len(vals) != 5:
            return None
        def num(x: str) -> float | int | None:
            try:
                return float(x)
            except ValueError:
                return None
        return {"clock_sm_mhz": num(vals[0]), "power_w": num(vals[1]),
                "memory_used_mib": num(vals[2]), "memory_free_mib": num(vals[3]),
                "utilization_pct": num(vals[4]), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    except (OSError, subprocess.SubprocessError):
        return None


def run_waiter() -> tuple[dict[str, Any], str]:
    cmd = [str(PYTHON), str(ROOT / "src/dual_target/gpu_wait.py"), "--project-root", str(ROOT), "--run-id", RUN_ID]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    state_path = ROOT / "outputs/dual_target_v1/dt1_gpu_wait_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state["waiter_exit_code"] = proc.returncode
    state["waiter_stdout_tail"] = proc.stdout[-2000:]
    state["waiter_stderr_tail"] = proc.stderr[-2000:]
    return state, " ".join(cmd)


def lock_is_released(lock_path: Path) -> bool:
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return True


def admission() -> dict[str, Any]:
    global _ACTIVE_LEASE
    state, wait_cmd = run_waiter()
    result: dict[str, Any] = {"run_id": RUN_ID, "waiter_command": wait_cmd, "waiter_state": state,
                              "admission_source": state.get("admission_source", "MISSING"),
                              "policy": state.get("admission_policy", "MISSING"),
                              "admission_snapshot": state.get("snapshot") or "MISSING", "lease_released": False}
    if state.get("waiter_exit_code") != 0 or state.get("status") != "GPU_READY_LOCKED_RECHECKED":
        result["status"] = "MISSING: gpu_wait did not reach live admission"
        result["reason"] = state.get("waiter_stderr_tail", "unknown")
        result["lease_released"] = lock_is_released(ROOT / "outputs/dual_target_v1/.dt1_gpu_wait.lock")
        return result
    lease = None
    try:
        from src.dual_target.gpu_wait import acquire_live_admission
        lease = acquire_live_admission(ROOT, run_id=RUN_ID, actual_argv=sys.argv)
        result["status"] = "admitted"
        result["receipt"] = lease.state
        result["fresh_snapshot"] = lease.snapshot.as_dict()
    except Exception as exc:  # noqa: BLE001 - report exact admission failure
        result["status"] = "MISSING: acquire_live_admission failed"
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Keep a successful lease through the complete measurement.  It is
        # closed by main() in its outer finally block.
        if lease is not None:
            _ACTIVE_LEASE = lease
        result["lease_released"] = False if lease is not None else lock_is_released(ROOT / "outputs/dual_target_v1/.dt1_gpu_wait.lock")
    return result


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_s": None, "p50_s": None, "p95_s": None, "samples_per_s": None}
    return {"mean_s": statistics.mean(values), "p50_s": statistics.median(values),
            "p95_s": statistics.quantiles(values, n=20, method="inclusive")[18] if len(values) > 1 else values[0],
            "samples_per_s": 1.0 / statistics.mean(values)}


def benchmark_path(path_name: str, batch_size: int, workers: int, model: Any) -> dict[str, Any]:
    """Measure frozen vision features plus the language-model CE head."""
    from datasets.navila_r2r import NavilaR2RDataset
    ds = NavilaR2RDataset(INDEX, FRAMES)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True)
    batch = next(iter(loader))
    pixels = batch["frames"].to(device="cuda", dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0
    # Text sequence is frozen at 512 image tokens + 160 language tokens.
    text_embeds = torch.randn(batch_size, 512 + 160, 960, device="cuda", dtype=torch.bfloat16)
    labels = torch.zeros(batch_size, 12, device="cuda", dtype=torch.long)
    times: list[float] = []
    clocks: list[dict[str, Any]] = []
    cached = None
    if path_name == "precomputed_cache":
        with torch.no_grad():
            cached = model.get_image_features(pixels).pooler_output.detach()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.no_grad():
            if path_name == "online_encoding":
                _ = model.get_image_features(pixels).pooler_output
            else:
                _ = cached
            core = model.get_base_model() if hasattr(model, "get_base_model") else model
            text_module = core.model.text_model if hasattr(core, "model") else core.text_model
            text_out = text_module(inputs_embeds=text_embeds, use_cache=False, return_dict=True)
            logits = model.lm_head(text_out.last_hidden_state[:, -12:])
            _ = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        sample = smi_sample()
        clocks.append(sample)
        if sample and isinstance(sample.get("memory_free_mib"), (int, float)) and sample["memory_free_mib"] < 2048:
            raise RuntimeError("MISSING: free GPU memory fell below 2048 MiB; measurement stopped")
        if step >= WARMUP_STEPS:
            times.append(elapsed)
    smi = smi_sample()
    smi_used = [s.get("memory_used_mib") for s in clocks if s and isinstance(s.get("memory_used_mib"), (int, float))]
    timing = stats(times)
    if timing["samples_per_s"] is not None:
        timing["samples_per_s"] = batch_size / timing["mean_s"]
    return {"batch_size": batch_size, "num_workers": workers, "path": path_name,
            "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS, "timing": timing,
            "torch_cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nvidia_smi_peak_memory_used_mib": max(smi_used) if smi_used else (smi.get("memory_used_mib") if smi else None),
            "gpu_samples": clocks, "status": "measured"}


def apply_frozen_training_config(model: Any) -> tuple[Any, int]:
    """Apply the v3 LoRA + lm_head freeze contract before probing."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"], task_type=TaskType.CAUSAL_LM,
        ))
    except Exception:
        # A missing PEFT install is reported by the caller; retaining the base
        # model here keeps the diagnostic useful for environments that only
        # provide transformers.
        for p in model.lm_head.parameters():
            p.requires_grad_(True)
    for name, p in model.named_parameters():
        if "vision_model" in name:
            p.requires_grad_(False)
    return model, sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    os.environ["HF_HOME"] = "/mnt/wxh/go2_short_vln/cache/huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report: dict[str, Any] = {
        "task_id": "P4-T6", "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_command": " ".join([str(PYTHON), str(Path(__file__))]),
        "inputs": {"index": str(INDEX), "index_sha256": sha256(INDEX) if INDEX.exists() else "MISSING",
                    "frames_root": str(FRAMES), "frames_sha256": "MISSING: directory aggregate not computed",
                    "model": str(MODEL), "model_sha256": "MISSING: directory aggregate not computed"},
        "config": {"tokenizer_max_length": 160, "pad_language_to": "longest", "history_frames": 8,
                   "tokens_per_frame": 64, "precision": "bf16", "lora_target_modules": ["q_proj", "v_proj"],
                   "lora_rank": 16, "lora_alpha": 32, "vision_encoder_frozen": True,
                   "lm_head_trainable": True, "tie_word_embeddings": False, "objective": "action_text next-token cross-entropy"},
        "gpu_admission": {}, "warmup": {"warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
                                         "clock_power_samples": [], "status": "MISSING"},
        "batch_sizes": [], "online_vs_precomputed": {},
        "vision_cache": {"feature_shape": [64, 960], "dtype": "bf16", "bytes_per_frame": FEATURE_BYTES,
                          "kib_per_frame": FEATURE_BYTES / 1024, "all_frames": TOTAL_FRAMES,
                          "total_bytes": TOTAL_FRAMES * FEATURE_BYTES,
                          "total_gib": TOTAL_FRAMES * FEATURE_BYTES / 1024**3,
                          "status": "representation-size measurement; encoder run MISSING if no CUDA"},
        "epoch_wall_clock_hours": {}, "missing": []}
    try:
        import pytest  # type: ignore # noqa: F401
    except ImportError:
        report["missing"].append("MISSING: pytest package unavailable in this interpreter; CPU test was invoked directly")
    report["gpu_admission"] = admission()
    try:
        if not torch.cuda.is_available() or report["gpu_admission"].get("status") != "admitted":
            report["missing"] += ["MISSING: CUDA/NVIDIA driver unavailable; no live throughput, memory, GPU utilization, GPU clock or power measurements",
                                   "MISSING: online encoding vs precomputed samples/s", "MISSING: maximum usable batch size",
                                   "MISSING: vision cache precompute wall-clock"]
            report["online_vs_precomputed"] = {"online_encoding_samples_per_s": "MISSING", "precomputed_cache_samples_per_s": "MISSING"}
            for bs in [1, 2, 4, 8, 16, 32, 64]:
                for workers in [0, 2]:
                    for path in ["online_encoding", "precomputed_cache"]:
                        report["batch_sizes"].append({"batch_size": bs, "num_workers": workers, "path": path,
                                                       "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
                                                       "timing": {"mean_s": "MISSING", "p50_s": "MISSING", "p95_s": "MISSING", "samples_per_s": "MISSING"},
                                                       "torch_cuda_max_memory_allocated_bytes": "MISSING",
                                                       "nvidia_smi_peak_memory_used_mib": "MISSING", "status": "MISSING"})
        else:
            try:
                from transformers import SmolVLMForConditionalGeneration
                model = SmolVLMForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
                                                                          torch_dtype=torch.bfloat16).to("cuda").eval()
                model, trainable_params = apply_frozen_training_config(model)
                report["config"]["trainable_parameters_observed"] = trainable_params
                stop_batch = False
                for bs in [1, 2, 4, 8, 16, 32, 64]:
                    if stop_batch:
                        break
                    for workers in [0, 2]:
                        for path in ["online_encoding", "precomputed_cache"]:
                            try:
                                row = benchmark_path(path, bs, workers, model)
                                report["batch_sizes"].append(row)
                                report["warmup"]["clock_power_samples"].extend([s for s in row["gpu_samples"] if s])
                            except RuntimeError as exc:
                                if "out of memory" in str(exc).lower():
                                    report["batch_sizes"].append({"batch_size": bs, "num_workers": workers, "path": path, "status": "OOM"})
                                    torch.cuda.empty_cache(); stop_batch = True; break
                                report["missing"].append(f"MISSING: benchmark {path} bs={bs} workers={workers}: {exc}")
                online = [r for r in report["batch_sizes"] if r.get("path") == "online_encoding" and r.get("status") == "measured"]
                cached = [r for r in report["batch_sizes"] if r.get("path") == "precomputed_cache" and r.get("status") == "measured"]
                report["online_vs_precomputed"] = {"online_encoding_samples_per_s": online[-1]["timing"]["samples_per_s"] if online else "MISSING",
                                                    "precomputed_cache_samples_per_s": cached[-1]["timing"]["samples_per_s"] if cached else "MISSING"}
                measured_bs = [r["batch_size"] for r in report["batch_sizes"] if r.get("status") == "measured"]
                report["maximum_usable_batch_size"] = max(measured_bs) if measured_bs else "MISSING"
                report["warmup"]["status"] = "measured"
            except Exception as exc:  # noqa: BLE001
                report["missing"].append(f"MISSING: local SmolVLM load/benchmark failed: {type(exc).__name__}: {exc}")
        report.setdefault("maximum_usable_batch_size", "MISSING")
        for mult in [1, 2, 3]:
            online_sps = report["online_vs_precomputed"].get("online_encoding_samples_per_s")
            cached_sps = report["online_vs_precomputed"].get("precomputed_cache_samples_per_s")
            report["epoch_wall_clock_hours"][str(mult)] = {
                "online_encoding": (TOTAL_SAMPLES * mult / float(online_sps) / 3600 if isinstance(online_sps, (int, float)) else "MISSING: samples/s unavailable"),
                "precomputed_cache": (TOTAL_SAMPLES * mult / float(cached_sps) / 3600 if isinstance(cached_sps, (int, float)) else "MISSING: samples/s unavailable"),
                "formula": f"{TOTAL_SAMPLES} samples × {mult} epoch ÷ {online_sps if isinstance(online_sps, (int, float)) else 'MISSING(samples_per_s)'} ÷ 3600 (online); "
                           f"{TOTAL_SAMPLES} samples × {mult} epoch ÷ {cached_sps if isinstance(cached_sps, (int, float)) else 'MISSING(samples_per_s)'} ÷ 3600 (cache)"
            }
    finally:
        if _ACTIVE_LEASE is not None:
            _ACTIVE_LEASE.close()
            report["gpu_admission"]["lease_released"] = lock_is_released(ROOT / "outputs/dual_target_v1/.dt1_gpu_wait.lock")
    out = ROOT / "reports/p4/t6_throughput.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "missing_count": len(report["missing"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
