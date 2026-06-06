# DSRL × LeRobot × Robosuite

**Steering a LeRobot Diffusion Policy with latent-space Reinforcement Learning (DSRL) on a custom Robosuite MimicGen task.**

This repository packages two things:

1. **`dsrl/`** — a checkout of the upstream [DSRL](https://diffusion-steering.github.io) ("Steering Your Diffusion Policy with Latent Space Reinforcement Learning", Wagenmaker et al., CoRL 2025) implementation, built on a fork of Stable-Baselines3 and the DPPO diffusion-policy codebase.
2. **`lerobot_integration/`** — an original integration layer that runs DSRL on top of a **LeRobot-trained Diffusion Policy** and a **custom Robosuite environment** (`CabinetBowlEnv`, a bowl pick-and-place into a drawer), bridging two otherwise-incompatible Python/dependency stacks over an in-process IPC bridge.

> **What is DSRL?** Instead of fine-tuning the weights of a diffusion policy, DSRL freezes the policy and runs RL (SAC) in the *initial-noise space* of the denoiser. The RL agent learns *which noise to sample* so that the (frozen) diffusion policy produces high-reward actions. This makes RL fine-tuning lightweight, sample-efficient, and safe for an expensive pretrained policy.

---

## Why an integration layer is needed

The base diffusion policy and the simulator live in **mutually incompatible dependency worlds**:

| | Simulator stack | Learning stack |
|---|---|---|
| Conda env | `mimicgen` | `lerobot` |
| Python | legacy (3.8/3.9-era) | modern (3.10+) |
| Key deps | Robosuite + MuJoCo + legacy `gym`, PyOpenGL/EGL | PyTorch, LeRobot, Gymnasium, Stable-Baselines3 (DSRL fork) |
| Holds | the custom `CabinetBowlEnv` | the trained DP + the SAC/DSRL agent |

They cannot share one interpreter. The integration solves this with a **decoupled client/server architecture over ZeroMQ**: the RL/policy process (`lerobot` env) drives N simulator processes (`mimicgen` env), one per parallel environment, talking over local TCP sockets with pickled messages.

```
                         lerobot conda env                                  mimicgen conda env
┌──────────────────────────────────────────────────────────┐        ┌────────────────────────────────┐
│  train_remote.py                                           │        │  zmq_server.py (port 5555)     │
│    SubprocVecEnv ── ZMQRemoteEnv(5555) ──TCP:5555────────────────▶ │    CabinetBowlEnv (Robosuite)  │
│                  ├─ ZMQRemoteEnv(5556) ──TCP:5556──────────────┐    │    + Base/ActionChunk wrappers │
│                  └─ ZMQRemoteEnv(...)  ──TCP:....───────────┐  │    └────────────────────────────────┘
│                                                            │  └─▶ zmq_server.py (port 5556) ...
│    LerobotDiffusionPolicyEnvWrapper (SAC noise ⇆ DP)       │  └───▶ zmq_server.py (port ....) ...
│      └─ LerobotDPPOPolicyWrapper → LeRobot DiffusionPolicy │
│    SAC (Stable-Baselines3 / DSRL fork)                     │
└──────────────────────────────────────────────────────────┘
```

The SAC agent picks a **noise vector**; `LerobotDiffusionPolicyEnvWrapper` feeds that noise to the frozen LeRobot DP, which denoises it into a **robot action chunk**; the chunk is shipped to a remote Robosuite process, executed, and the resulting reward/observation flows back to SAC.

---

## Repository layout

```
DSRL/
├── README.md                 ← you are here (repo overview)
├── docs/
│   ├── SETUP.md              ← environments, custom Robosuite, LeRobot, checkpoint, configurable paths
│   └── TRAINING.md           ← running training, architecture deep-dive, interface contract, troubleshooting
│
├── dsrl/                      ← upstream DSRL (CoRL 2025) — the algorithm + SB3/DPPO forks
│   ├── train_dsrl.py         ← upstream Robomimic/Gym entry point
│   ├── env_utils.py          ← upstream DiffusionPolicyEnvWrapper reference
│   ├── cfg/                  ← Hydra configs (robomimic/, gym/)
│   ├── dppo/                 ← DPPO diffusion-policy submodule (fork)
│   └── stable-baselines3/    ← Stable-Baselines3 submodule with DSRL/SAC noise algorithms (fork)
│
└── lerobot_integration/      ← THIS PROJECT: DSRL over a LeRobot DP + custom Robosuite env
    ├── train_remote.py       ← main training entry point (run this)
    ├── infer_remote.py       ← load a trained SAC checkpoint and roll out
    ├── zmq_client.py         ← Gymnasium-side proxy; spawns a server subprocess per env
    ├── zmq_server.py         ← Robosuite-side server (runs in the mimicgen env)
    ├── env_wrapper.py        ← obs/action formatting + SAC-noise VecEnv wrapper
    ├── policy_wrapper.py     ← wraps the LeRobot DiffusionPolicy for DSRL noise steering
    ├── debug_single_env.py   ← single-env, no-SAC mechanism debugger
    ├── debug_normalization.py← normalization-pipeline checker (no Robosuite needed)
    └── README.md             ← original component-level notes
```

---

## Quick start

Full, copy-pasteable instructions are in **[docs/SETUP.md](docs/SETUP.md)** and **[docs/TRAINING.md](docs/TRAINING.md)**. The short version:

```bash
# 1. Set up the two conda environments and the custom Robosuite + LeRobot DP checkpoint
#    → see docs/SETUP.md

# 2. Edit the machine-specific paths flagged in docs/SETUP.md § "Paths you must change"

# 3. Launch training from inside the lerobot env
conda activate lerobot
cd lerobot_integration
python train_remote.py algorithm=dsrl_sac use_wandb=false
```

This auto-aligns the action space to the checkpoint, spawns one Robosuite server subprocess per parallel env, and runs SAC in the DP's noise space. Checkpoints land in `lerobot_integration/logs/checkpoints/`.

> ⚠️ **Before you run anything**, read **[docs/SETUP.md § Paths you must change](docs/SETUP.md#paths-you-must-change)**. The integration scripts contain absolute paths (the DP checkpoint, the `mimicgen` Python interpreter, the custom Robosuite source tree) that are specific to the original machine and *must* be edited for your setup.

---

## Documentation map

| Document | Read it when you want to… |
|---|---|
| **[docs/SETUP.md](docs/SETUP.md)** | install dependencies, build the two conda envs, register the custom Robosuite task, get the DP checkpoint, and fix machine-specific paths |
| **[docs/TRAINING.md](docs/TRAINING.md)** | run training, understand the IPC architecture, learn the obs/action **interface contract** (the correctness-critical part), tune hyperparameters, and troubleshoot |
| **[lerobot_integration/README.md](lerobot_integration/README.md)** | read the original per-component notes (kept for reference) |
| **[dsrl/README.md](dsrl/README.md)** | run *upstream* DSRL on standard Robomimic/Gym tasks |

---

## License & citation

The `dsrl/` subtree (and its `dppo/` and `stable-baselines3/` submodules) is licensed by its upstream authors — see the headers and `LICENSE` files within those trees. The `lerobot_integration/` layer is the contribution of this repository.

If you use DSRL, please cite the original paper:

```bibtex
@article{wagenmaker2025steering,
  author  = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  title   = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal = {Conference on Robot Learning (CoRL)},
  year    = {2025},
}
```

- Paper: https://arxiv.org/pdf/2506.15799 · Project page: https://diffusion-steering.github.io
- LeRobot: https://github.com/huggingface/lerobot · Robosuite: https://robosuite.ai · MimicGen: https://mimicgen.github.io
