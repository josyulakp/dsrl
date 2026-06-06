# Setup Guide

This guide gets you from a clean machine to a runnable DSRL-over-LeRobot training job on the custom `CabinetBowlEnv` Robosuite task.

There are **two conda environments** by design (see the [README](../README.md#why-an-integration-layer-is-needed) for why):

| Env | Holds | Runs |
|---|---|---|
| `mimicgen` | the custom Robosuite `CabinetBowlEnv`, MuJoCo, legacy `gym` | `zmq_server.py` (one subprocess per parallel env) |
| `lerobot`  | PyTorch, LeRobot Diffusion Policy, Gymnasium, SB3/DSRL fork | `train_remote.py`, the SAC agent, the policy wrapper |

The `lerobot` process is the orchestrator; it spawns `mimicgen` subprocesses automatically. You launch training from the `lerobot` env only.

---

## 0. Prerequisites

- **OS:** Linux (the IPC bridge uses `bash -c "source .../activate ..."` and EGL headless rendering).
- **GPU:** NVIDIA GPU with CUDA (the DP runs DDIM denoising every env step; CPU is impractical).
- **Conda/Miniconda** installed (the default examples assume `/opt/miniconda3`).
- **Disk:** the DP checkpoint is ~1 GB; replay buffers and checkpoints grow during training.
- Headless rendering uses MuJoCo's **EGL** backend (`MUJOCO_GL=egl`), so no physical display is required.

---

## 1. Get the code

```bash
# The dsrl/ subtree uses git submodules (the SB3 + DPPO forks).
cd /path/to/DSRL/dsrl
git submodule update --init --recursive
```

`dsrl/` already points at the upstream forks:
- `stable-baselines3` → `ajwagen/stable-baselines3-dsrl` (adds the `DSRL`/`SACDiffusionNoise` algorithms)
- `dppo` → `ajwagen/dppo-dsrl`

---

## 2. The `lerobot` environment (orchestrator)

This env runs the RL agent and the diffusion policy.

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot

# --- LeRobot (provides DiffusionPolicy + the processor/normalization pipeline) ---
pip install lerobot            # or: pip install -e /path/to/your/lerobot checkout

# --- DSRL's Stable-Baselines3 fork (provides SAC + the DSRL noise algorithm) ---
cd /path/to/DSRL/dsrl/stable-baselines3
pip install -e .

# --- Misc deps used by the integration layer ---
pip install pyzmq hydra-core omegaconf opencv-python wandb
```

> **LeRobot version note.** `policy_wrapper.py` imports:
> - `lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy` and `_make_noise_scheduler`
> - `lerobot.policies.factory.make_pre_post_processors`
>
> These are the **modern** LeRobot APIs where normalization lives in a *separate* pre/post-processor pipeline (not inside `generate_actions()`). If your LeRobot is older and lacks `make_pre_post_processors`, the policy will receive un-normalized inputs and the base policy will be effectively broken — see [TRAINING.md § Interface contract](TRAINING.md#interface-contract-correctness-critical). Pin/checkout a LeRobot revision that exposes these symbols.

---

## 3. The `mimicgen` environment (simulator)

This env holds the custom Robosuite task. It is the same env used to generate the demonstration data the DP was trained on, which is what keeps the simulator and the policy in sync.

```bash
conda create -n mimicgen python=3.8 -y
conda activate mimicgen

# Robosuite + MuJoCo + legacy gym + MimicGen, per your MimicGen/Robosuite install.
# The custom CabinetBowlEnv lives in your robosuite source tree at:
#   <mimicgen>/robosuite/robosuite/environments/manipulation/cabinet_bowl_env.py
pip install -e /path/to/mimicgen/robosuite     # custom Robosuite (editable)
pip install pyzmq                               # the server speaks ZeroMQ
```

The server registers the custom env at import time:

```python
# zmq_server.py
sys.path.append("/mnt/DataDrive/samaksh/mimicgen/robosuite")   # ← machine-specific (see §5)
from robosuite.environments.manipulation.cabinet_bowl_env import CabinetBowlEnv
register_env(CabinetBowlEnv)
```

### The custom task & robot

`zmq_server.py` constructs the env with settings that **must** match the DP's training data:

- **Env:** `CabinetBowlEnv` — pick a bowl and place it into a drawer.
- **Robot:** `PR_V4P1SR_UMI` (a UMI-gripper robot). The UMI gripper is what exposes the `gripper0_eye_in_hand` wrist camera the DP was trained on. **Not** the default `Panda`.
- **Controller:** `JOINT_POSITION` with `kp=150`, `output_max/min=±0.08` — matching the data-generation config. The controller is started from the Robosuite default (so required fields like `interpolation` exist) and only those values are overridden.
- **Cameras:** `topview_custom`, `up_sideview`, `gripper0_eye_in_hand`, each `480×640`.
- **`control_freq=20`**.

> Confirm the robot model `PR_V4P1SR_UMI` and the `CabinetBowlEnv` are both registered/importable in *your* Robosuite tree. These are custom assets; they are not part of stock Robosuite.

---

## 4. The base Diffusion Policy checkpoint

DSRL **steers a frozen, already-trained** LeRobot Diffusion Policy — it does not train one. You need the DP checkpoint that was trained on the `CabinetBowlEnv` bowl-pick-and-place data.

The reference checkpoint used by the scripts is a standard LeRobot pretrained-model directory:

```
.../dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model/
├── config.json
├── model.safetensors                                       (~1 GB)
├── train_config.json
├── policy_preprocessor.json   + ..._normalizer_processor.safetensors
└── policy_postprocessor.json  + ..._unnormalizer_processor.safetensors
```

The `policy_*processor*` files are **essential** — they carry the normalization statistics (`STATE`/`ACTION` → MIN_MAX, `VISUAL` → MEAN_STD). `load_lerobot_policy()` loads them via `make_pre_post_processors`. Without them the DP is broken.

**Checkpoint spec** (from `config.json`, auto-detected at runtime — you don't set these by hand):

| Field | Value | Meaning |
|---|---|---|
| `n_obs_steps` | 2 | observation history fed to the DP |
| `horizon` | 16 | UNet prediction horizon |
| `n_action_steps` | 8 | executed action chunk = DSRL noise window |
| action `shape` | `[8]` | 7 joint targets + 1 gripper |
| input `observation.state` | `[8]`, `observation.eef_state` | `[6]` | low-dim state |
| input images | `left`, `right`, `wrist`, each `[3,480,640]` | camera obs |
| `noise_scheduler_type` | `DDPM` → overridden to **`DDIM`** | DSRL requires DDIM sampling |
| `num_train_timesteps` | 100 | denoising steps |

`policy_wrapper.py` overrides the scheduler to DDIM at load time (DSRL needs a deterministic, noise-controllable sampler).

---

## 5. Paths you must change

The integration scripts hardcode absolute paths from the original machine. **Edit these before running**, or the scripts will fail or silently load the wrong assets.

| File · line | Hardcoded value | Change to |
|---|---|---|
| `lerobot_integration/train_remote.py` (`model_path`) | `/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model` | path to **your** DP `pretrained_model/` dir |
| `lerobot_integration/infer_remote.py` (`model_path`) | same as above | same — keep in sync with training |
| `lerobot_integration/zmq_server.py` (`sys.path.append`) | `/mnt/DataDrive/samaksh/mimicgen/robosuite` | path to **your** custom Robosuite source tree (the one containing `cabinet_bowl_env.py`) |
| `lerobot_integration/zmq_client.py` (server launch) | `source /opt/miniconda3/bin/activate mimicgen` | your conda activate path + env name (or use `MIMICGEN_PYTHON`, below) |
| `lerobot_integration/train.py` (`sys.path.append`, `model_path`) | same as `zmq_server`/`train_remote` | the legacy single-env variant — only if you use it |

### Recommended: `MIMICGEN_PYTHON` instead of editing `zmq_client.py`

`zmq_client.py` honors a `MIMICGEN_PYTHON` environment variable that points directly at the simulator interpreter, bypassing the `conda activate` shell line entirely:

```bash
export MIMICGEN_PYTHON=/opt/miniconda3/envs/mimicgen/bin/python
```

When set, the client launches the server as `"$MIMICGEN_PYTHON zmq_server.py ..."`. This is more robust than relying on `conda activate` being available in the launching shell. Prefer this over editing the activate path.

---

## 6. Verify the setup before full training

Two debug scripts let you validate the stack incrementally (run from `lerobot_integration/`, in the `lerobot` env):

```bash
# (a) Normalization pipeline only — does NOT need Robosuite. Confirms the DP
#     pre/post-processors load and produce sane normalized/un-normalized ranges.
python debug_normalization.py

# (b) One real Robosuite env + the real DP, no SAC. Drives the full loop
#     (noise → DP → robot action → env → reward) for a few steps and prints every tensor range.
python debug_single_env.py             # random noise-space actions (RL-like)
python debug_single_env.py --gaussian  # DP's natural N(0,1) prior → should produce purposeful motion & reward > 0
```

If `--gaussian` produces purposeful arm motion and occasional reward, the obs/action **interface contract** is correct and you are ready to train. If the arm holds still or thrashes, re-check [TRAINING.md § Interface contract](TRAINING.md#interface-contract-correctness-critical) before burning GPU hours on SAC.

→ Continue to **[TRAINING.md](TRAINING.md)**.
