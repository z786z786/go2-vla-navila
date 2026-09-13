# M0 Environment Audit

Audit target: `wxh@202.199.13.104` (`neurt-Precision-7920-Tower`)  
Audit timestamp: `2026-08-30T13:18:10+08:00`  
Scope: read-only environment inspection plus this report; no installation or system modification.

## Guardrails Applied

- No NVIDIA driver changes.
- No system CUDA changes.
- No package, Conda environment, NaVILA-Bench, LeRobot, or SmolVLA installation.
- No existing environment, project, model, process, or checkpoint was modified or removed.
- M0 stops after the two reports are created and verified.

## Hardware

| Item | Observed value |
|---|---|
| Host | `neurt-Precision-7920-Tower` |
| Architecture | `x86_64` |
| CPU | 2 × Intel Xeon Gold 6226R @ 2.90 GHz |
| CPU topology | 2 sockets, 16 cores/socket, 2 threads/core, 64 logical CPUs |
| RAM | 503 GiB total, 486 GiB available at audit time |
| Swap | 2.0 GiB, unused |
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24576 MiB total |

## GPU and CUDA

| Layer | Observed value |
|---|---|
| NVIDIA driver | `580.173.02` |
| Driver-supported CUDA version reported by `nvidia-smi` | `13.0` |
| System CUDA symlink | `/usr/local/cuda -> /usr/local/cuda-12.2` |
| CUDA toolkit / `nvcc` | `12.2`, build `V12.2.91` |
| System CUDA runtime | `libcudart.so.12` from CUDA 12.2 |
| PyTorch CUDA availability | Available in both audited ML environments |

Dynamic snapshot at `2026-08-30 13:18:10 +08:00`:

- VRAM used: 9623 MiB
- VRAM free: 14493 MiB
- GPU utilization: 0%
- Temperature: 52 °C
- Performance state: P8
- Other users had two Python compute processes using 6692 MiB and 2859 MiB. These processes were not touched.

The CUDA version displayed by `nvidia-smi`, the installed system toolkit, and the CUDA runtime bundled with each PyTorch build are separate compatibility layers and must not be treated as the same version.

## Operating System and Storage

| Item | Observed value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | `6.8.0-136-generic` |
| System Python | `/usr/bin/python3`, Python 3.10.12; no unversioned `python` command |
| Root filesystem | `/dev/sda2`, ext4, 916 GiB total, 818 GiB used, 52 GiB free, 95% used |
| Data filesystem | `/dev/sdb1`, ext4, mounted at `/mnt`, 3.6 TiB total, 1.4 TiB used, 2.1 TiB free, 41% used |
| `/workspace` | Not present |
| `/data` | Not present |

The 52 GiB remaining on `/` is not suitable for the future simulator assets, datasets, Conda package cache, or checkpoints. M1 must report download sizes first and place large content on `/mnt` or use documented links from the project tree.

## Conda, Python, and PyTorch

Conda is installed at `/home/wxh/miniconda3` and reports version `26.1.1`. `mamba` and `micromamba` were not found. Conda reported only the following environments:

| Environment | Path | Python | PyTorch | PyTorch CUDA runtime | `torch.cuda.is_available()` |
|---|---|---:|---:|---:|---:|
| `base` | `/home/wxh/miniconda3` | 3.13.12 | Not installed | N/A | N/A |
| `llada` | `/home/wxh/miniconda3/envs/llada` | 3.10.20 | 2.6.0+cu124 | 12.4 | True, RTX 3090 detected |
| `openvla` | `/home/wxh/miniconda3/envs/openvla` | 3.10.20 | 2.2.0+cu121 | 12.1 | True, RTX 3090 detected |

No existing Conda environment is named `navila-isaac` or `smolvla`. None of the three audited environments contains the `lerobot` package.

## Existing Robotics and VLA Installations

| Component | Status | Evidence |
|---|---|---|
| Isaac Sim | Installed | `/home/wxh/isaacsim`, version `5.1.0-rc.19+release.26219.9c81211b.gl` |
| Isaac Lab | Installed | `/home/wxh/IsaacLab`, `VERSION` = `2.3.1`; `_isaac_sim` links to `/home/wxh/isaacsim` |
| Isaac Lab provenance | Incomplete | Directory is not a Git checkout, so its exact source commit cannot be recovered from Git metadata |
| NaVILA-Bench | Not found | No matching project directory under `/home/wxh` or `/mnt` within the audited search depth |
| LeRobot | Not found | No matching project directory and no importable `lerobot` package in the reported Conda environments |
| SmolVLA | Not found | No matching project directory and no importable `lerobot.policies.smolvla` package |

Other unrelated robotics/VLA projects and existing Isaac installations were observed but not inspected beyond what M0 required.

## Upstream Compatibility Facts

The NaVILA-Bench README was checked on 2026-08-30:

- Ubuntu 20.04 or newer.
- Python 3.10.
- Isaac Sim 4.1.0.
- Tested with Isaac Lab 1.1.0 and instructs users to use the modified fork at `https://github.com/yang-zj1026/IsaacLab`.
- Source: <https://github.com/yang-zj1026/NaVILA-Bench>

The current LeRobot `main` `pyproject.toml` was checked on 2026-08-30:

- Project version 0.6.2.
- Python `>=3.12`.
- PyTorch `>=2.7,<2.12.0`.
- SmolVLA is provided through the `smolvla` optional dependency group.
- Source: <https://github.com/huggingface/lerobot/blob/main/pyproject.toml>

## Recommended Isolated Environments

### `navila-isaac`

- Create only in M1 after re-reading the checked-out NaVILA-Bench README and source.
- Use Python 3.10, Isaac Sim 4.1.0, and the NaVILA-maintained modified Isaac Lab 1.1.0 line.
- Do not replace or repurpose `/home/wxh/isaacsim` or `/home/wxh/IsaacLab`; they are Isaac Sim 5.1 / Isaac Lab 2.3.1 and do not match NaVILA-Bench's tested stack.
- Do not downgrade the NVIDIA driver or replace system CUDA. Any compatibility issue must first be reproduced and diagnosed in the isolated environment.

### `smolvla`

- Create only in the later LeRobot milestone, separate from `navila-isaac`.
- Start from Python 3.12 for the currently audited LeRobot main line.
- Pin a specific LeRobot release or commit at installation time and record it; do not rely on a moving `main` branch.
- Keep its PyTorch and CUDA wheel dependencies isolated from Isaac Sim.

## Compatibility Risks

1. **Root disk pressure:** `/` is 95% used with only 52 GiB free. Large downloads must not target default home/Conda locations without a storage plan.
2. **NaVILA version mismatch:** the installed Isaac Sim 5.1 / Isaac Lab 2.3.1 stack is substantially newer than NaVILA-Bench's documented Isaac Sim 4.1 / Isaac Lab 1.1 stack.
3. **Shared GPU occupancy:** 9623 MiB was already allocated by other users during M0. Isaac rendering and SmolVLA training must be scheduled separately and only after confirming available VRAM.
4. **Isaac Lab provenance:** the installed `/home/wxh/IsaacLab` is not a Git checkout, so local modifications or its exact source commit cannot be audited through Git.
5. **Moving LeRobot requirements:** current main requires Python 3.12 and PyTorch 2.7+, unlike the existing Python 3.10 ML environments. A dedicated environment is mandatory.
6. **CUDA version layers:** driver-supported CUDA version 13.0, system toolkit 12.2, and PyTorch CUDA 12.1/12.4 coexist. Global CUDA changes would risk unrelated workloads and are prohibited.
7. **SSH alias mismatch:** the runbook says `ssh neu3070`, while local SSH configuration defines `neu-3070` for the same server. Successful audit commands used `wxh@202.199.13.104` directly.

## Commands Executed

All commands below were read-only except the final creation of `/home/wxh/go2_short_vln/reports` and its two Markdown files.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'printf "M0_SSH_OK\n"; hostname; whoami; pwd'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'cat /etc/os-release; uname -a; lscpu; free -h; df -hT; df -ih'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,pstate --format=csv,noheader; command -v nvcc || true; nvcc --version 2>/dev/null || true; ls -ld /usr/local/cuda* 2>/dev/null || true; command -v nvidia-container-cli || true'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'command -v python || true; command -v python3 || true; python --version 2>&1 || true; python3 --version 2>&1 || true; command -v conda || true; command -v mamba || true; command -v micromamba || true; find /home/wxh -maxdepth 4 -type f \( -name conda -o -name mamba -o -name micromamba \) 2>/dev/null | head -40'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 '/home/wxh/miniconda3/bin/conda --version; /home/wxh/miniconda3/bin/conda env list; /home/wxh/miniconda3/bin/conda config --show envs_dirs; ls -1 /home/wxh/miniconda3/envs'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 '/usr/local/cuda/bin/nvcc --version 2>&1 || true; readlink -f /usr/local/cuda 2>/dev/null || true; ldconfig -p 2>/dev/null | grep libcudart | head -20 || true'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'printf "IsaacSim VERSION: "; cat /home/wxh/isaacsim/VERSION 2>/dev/null || true; printf "IsaacLab VERSION: "; cat /home/wxh/IsaacLab/VERSION 2>/dev/null || true; grep -nE "Isaac Sim|IsaacSim|Python" /home/wxh/IsaacLab/README.md | head -40; ls -ld /home/wxh/IsaacLab/_isaac_sim; readlink -f /home/wxh/IsaacLab/_isaac_sim'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 '/home/wxh/miniconda3/bin/conda run -p /home/wxh/miniconda3 python -c "import sys; print(sys.version); import torch; print(torch.__version__)" 2>&1 || true'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 '/home/wxh/miniconda3/bin/conda run -p /home/wxh/miniconda3/envs/llada python -c "import sys; print(sys.version); import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")"'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 '/home/wxh/miniconda3/bin/conda run -p /home/wxh/miniconda3/envs/openvla python -c "import sys; print(sys.version); import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")"'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'find /home/wxh /mnt -maxdepth 3 -type d \( -iname "NaVILA-Bench" -o -iname "lerobot" -o -iname "smolvla" \) 2>/dev/null | sort'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'date --iso-8601=seconds; uname -r; free -h; df -hT / /mnt; nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true'
ssh -o BatchMode=yes -o ConnectTimeout=12 wxh@202.199.13.104 'mkdir -p /home/wxh/go2_short_vln/reports'
scp -p /private/tmp/go2_m0_stage/ENVIRONMENT.md /private/tmp/go2_m0_stage/MILESTONES.md wxh@202.199.13.104:/home/wxh/go2_short_vln/reports/
```

Upstream documentation was inspected at the two URLs listed in the compatibility section. No repository was cloned.

## M0 Acceptance Criteria

- [x] RTX 3090 detected.
- [x] `nvidia-smi` works normally.
- [x] NVIDIA driver and CUDA layers are explicitly documented.
- [x] Root and data-disk capacity are explicitly documented.
- [x] NVIDIA driver was not modified.
- [x] System CUDA was not replaced or modified.
- [x] Existing Isaac, Conda, and user workloads were not modified.
- [x] `reports/ENVIRONMENT.md` created.

M0 is complete. Do not begin M1 until the user sends `CONTINUE M1`.
