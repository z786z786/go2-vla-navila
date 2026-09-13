"""Portable collection, quality review, and study-configuration entry points."""
import argparse
import copy
import json
from pathlib import Path
from .contracts import ActionFilter, SCHEMA_VERSION, validate_episode
from .data import audit, read_episodes, summarize
from .runtime import ColorBoxPolicy, ExternalEnvironment, ModelPolicy, TwoBoxMock


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run(args):
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.backend != "mock" and not args.runtime_factory:
        raise ValueError("external backend requires --runtime-factory module:callable")
    if args.policy == "mock" and args.backend != "mock":
        raise ValueError("mock color policy is only supported with mock runtime")
    if args.policy != "mock" and (not args.model_factory or not args.checkpoint):
        raise ValueError("model policy requires --model-factory and --checkpoint")
    policy = (ColorBoxPolicy(args.stop_mode) if args.policy == "mock" else
              ModelPolicy(args.model_factory, args.checkpoint, args.history, args.state, args.denoise_steps))
    episodes = []
    for i in range(args.episodes):
        name = f"episode_{i:04d}"
        env = (TwoBoxMock(output/name) if args.backend == "mock" else
               ExternalEnvironment(args.backend, args.runtime_factory, args.enable_robot))
        control = ActionFilter(clipping=not args.no_clipping, slew=not args.no_slew)
        steps = []
        try:
            policy.reset()
            obs = env.reset("go to the red box" if i % 2 == 0 else "go to the blue box")
            for tick in range(args.max_steps):
                action = control.apply(policy.predict(obs), args.dt)
                row = obs.to_dict()
                frame = Path(row["rgb_path"]).resolve()
                if not frame.is_relative_to(output):
                    # Materialize external images so records survive checkout relocation.
                    import shutil
                    dest = output/name/f"frame_{tick:04d}{frame.suffix}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(frame, dest)
                    frame = dest
                row["rgb_path"] = str(frame.relative_to(output))
                steps.append({"observation": row, "action": action.to_dict()})
                if action.stop:
                    env.step(action)
                    break
                obs = env.step(action)
        finally:
            env.close()
        record = {"schema_version": SCHEMA_VERSION, "episode_id": name,
                  "scene_id": "two_box_mock" if args.backend == "mock" else args.backend,
                  "source": {"mock":"mock", "isaacsim":"simulation", "unitree":"real"}[args.backend],
                  "policy": args.policy, "outcome":"stopped" if steps[-1]["action"]["stop"] else "timeout",
                  "steps":steps}
        validate_episode(record)
        episodes.append(record)
    write_json(output/"episodes.json", {"episodes": episodes})
    write_json(output/"diagnostics.json", summarize(episodes))
    print(output/"episodes.json")


def review(args):
    source = Path(args.input).resolve()
    rejected = json.loads(Path(args.review).read_text()).get("reject", []) if args.review else []
    kept, issues = audit(read_episodes(source), source.parent, rejected)
    destination = Path(args.output).resolve()
    if destination == source:
        raise ValueError("review must not overwrite source")
    # Keep frame paths valid if the cleaned manifest is written elsewhere.
    import os
    kept = copy.deepcopy(kept)
    for episode in kept:
        for step in episode["steps"]:
            obs = step["observation"]
            obs["rgb_path"] = os.path.relpath(source.parent/obs["rgb_path"], destination.parent)
    write_json(destination, {"episodes": kept, "rejected":issues})
    print(json.dumps({"kept":len(kept), "rejected":len(issues)}))
    if issues and args.strict:
        raise SystemExit(1)


def study(args):
    baseline = {"expert":"continuous", "clipping":True, "slew":True,
                "history":1, "state":False, "stop_mode":"hysteresis",
                "denoise_steps":10, "model":"smolvla"}
    variants = [{"name":"baseline", "config":baseline}]
    for key, values in {"expert":["legacy"], "history":[4,8], "state":[True],
                        "stop_mode":["explicit"], "denoise_steps":[1,4,20], "model":["llada-v"]}.items():
        for value in values:
            variants.append({"name":f"{key}_{value}", "config":{**baseline,key:value}})
    for clipping, slew in [(False,False),(False,True),(True,False)]:
        variants.append({"name":f"clipping_{clipping}_slew_{slew}",
                         "config":{**baseline,"clipping":clipping,"slew":slew}})
    write_json(args.output, {"kind":"study_configuration", "variants":variants})
    print(args.output)


def main():
    parser = argparse.ArgumentParser(prog="go2-nav")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--backend", choices=["mock","isaacsim","unitree"], default="mock")
    p.add_argument("--policy", choices=["mock","smolvla","llada-v"], default="mock")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--output", default="artifacts/two_box")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--runtime-factory")
    p.add_argument("--model-factory")
    p.add_argument("--checkpoint")
    p.add_argument("--enable-robot", action="store_true")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--state", action="store_true")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--stop-mode", choices=["explicit","hysteresis"], default="hysteresis")
    p.add_argument("--no-clipping", action="store_true")
    p.add_argument("--no-slew", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("review")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--review")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=review)
    p = sub.add_parser("study")
    p.add_argument("--output", default="artifacts/study.json")
    p.set_defaults(func=study)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError,TypeError,FileNotFoundError,ImportError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
