# M1 Recovery and Current Gate

Date: 2026-08-30

## Network recovery

| Item | Official source | Required remote location | Status |
|---|---|---|---|
| NaVILA-Bench source at `e9d2db12ce5788c0f987d734c0094100b6bc0d3a` | https://github.com/yang-zj1026/NaVILA-Bench | `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench` | Recovered and checked out through project-local proxy |
| Modified Isaac Lab source at `4d558ec83878c4892a46591c85ba91ac9d3c1834` | https://github.com/yang-zj1026/IsaacLab | `/mnt/wxh/go2_short_vln/third_party/IsaacLab` | Recovered and checked out through project-local proxy |
| Matterport scene archive (~5.02 GB) and VLN-CE metadata | https://huggingface.co/datasets/Zhaojing/VLN-CE-Isaac | `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/` | Not yet downloaded; public asset download follows EULA acceptance and extension setup |

## Evidence and current gate

- A local HTTP proxy at `127.0.0.1:7897` was connected with a persistent SSH remote forward that listens only at remote `127.0.0.1:17997`.
- Project-local proxy variables were used for all source/package traffic; no system or global Git/Conda proxy setting was changed.
- GitHub, Hugging Face, and NVIDIA PyPI returned valid responses through the tunnel; both source revisions above were cloned successfully.
- The isolated Isaac Sim 4.1 package stack installed successfully. Its first import asks the operator to accept the NVIDIA Omniverse Kit EULA. This is an explicit authorization gate, not a package or network failure.

## Resume after explicit EULA acceptance

After the operator explicitly accepts the NVIDIA Omniverse Kit EULA, use the project-local environment variable only for M1 processes:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Then continue with the modified Isaac Lab supported installation path, bundled `rsl_rl`, the public Matterport asset download to the exact `/mnt` location above, and the official one-environment headless planner command. Do not use `/home/wxh/isaacsim` or `/home/wxh/IsaacLab`.
