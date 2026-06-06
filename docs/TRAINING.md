# Training Guide & Architecture

How to run DSRL training on `CabinetBowlEnv`, how the pieces fit together, and — most importantly — the **interface contract** that keeps the simulator, the diffusion policy, and the RL agent mutually consistent.

> Assumes you have completed **[SETUP.md](SETUP.md)**, including editing the machine-specific paths and verifying `debug_single_env.py --gaussian`.

---

## 1. Running training

From the `lerobot` conda env, inside `lerobot_integration/`:

```bash
conda activate lerobot
cd lerobot_integration

# Image-based DP → use the SAC variant (see §6 for why dsrl_na is not available here)
python train_remote.py algorithm=dsrl_sac use_wandb=false
```

Common overrides (Hydra CLI syntax, `key=value`):

```bash
# Enable Weights & Biases logging
python train_remote.py algorithm=dsrl_sac use_wandb=true

# Open an on-screen GUI window for env 0 only (needs a display; never opens one window per env)
python train_remote.py algorithm=dsrl_sac render=true

# Change the number of parallel simulator processes (also changes ports used)
python train_remote.py algorithm=dsrl_sac env.n_envs=4

# Change episode length (env steps before truncation)
python train_remote.py algorithm=dsrl_sac env.max_episode_steps=300
```

### What happens on launch

1. Loads the **frozen** LeRobot Diffusion Policy from `model_path` and its normalization processors.
2. **Auto-aligns** `cfg.action_dim` and `cfg.act_steps` to the checkpoint (`8` and `8`) — you never hand-set these; they always follow the checkpoint.
3. Spawns `env.n_envs` Robosuite **server subprocesses** in the `mimicgen` env, one per env, on ports `5555, 5556, …`. Each boots MuJoCo in headless EGL mode (the client sleeps ~8 s per server while it spins up).
4. Wraps the vectorized envs with `LerobotDiffusionPolicyEnvWrapper` so SAC sees the **noise space**.
5. Builds the SAC agent (3×2048 MLP actor & critic, 2 critics, Tanh) and runs `model.learn(total_timesteps=2_000_000)`.

### Outputs

| What | Where |
|---|---|
| Periodic SAC checkpoints | `lerobot_integration/logs/checkpoints/` (every `save_freq=1000` steps) |
| W&B run (if enabled) | project `dsrl` (from `cfg.wandb.project`), run name `lerobot_dsrl_cabinet_bowl_zmq` |
| Server stdout | inline (each server prints `[ZMQ Server @ <port>] …`) |

To stop cleanly, `Ctrl-C`; `ZMQRemoteEnv.close()` sends a `close` command and terminates each server subprocess.

---

## 2. Data flow (one SAC environment step)

```
SAC.actor ─▶ noise action a ∈ ℝ^(n_envs × 64)         # 64 = act_steps(8) × action_dim(8), bounded by ±action_magnitude
        │
        ▼   LerobotDiffusionPolicyEnvWrapper.step_async()           [lerobot env]
        ├─ reshape a → (n_envs, 8, 8)                               # the DSRL "initial noise"
        ├─ build DP batch from obs queues (n_obs_steps=2 history)   # state + eef_state + 3 cameras
        ├─ LerobotDPPOPolicyWrapper(batch, initial_noise):
        │     • preprocessor → normalize obs (MIN_MAX state, MEAN_STD images)
        │     • overlay the 8-step noise into the 16-step UNet horizon at slice [1:9]
        │     • DDIM generate_actions → normalized actions
        │     • postprocessor → un-normalize → 8-D ABSOLUTE robot targets per step
        └─ venv.step_async(diffused_actions)  → (n_envs, 64) absolute action chunks
        │
        ▼   ZMQRemoteEnv.step()  →  TCP  →  zmq_server.py            [mimicgen env, ×n_envs]
        ├─ LerobotActionChunkWrapper: reshape (64,) → (8, 8), run 8 sub-steps
        │     per sub-step  LerobotBaseEnvWrapper.step():
        │       • absolute joint target → JOINT_POSITION delta:  clip((target[:7]−cur_joint)/0.08, −1, 1)
        │       • gripper target [0,0.04] → cmd:  clip(1 − 2·target[7]/0.04, −1, 1)
        │       • env.step(ctrl); accumulate dense reward
        └─ return final obs, summed reward, done, info
        │
        ▼   step_wait()                                             [lerobot env]
        ├─ flatten terminal_observation dict → state (SB3 expects flat state on auto-reset)
        ├─ push new obs into the per-env history queues
        └─ return observation.state (8-D) + reward + done to SAC
```

The RL agent only ever sees the **8-D robot state** as observation and acts in the **64-D noise space**. Everything between (the diffusion denoising, the absolute→delta conversion, the simulator) happens inside the wrappers.

---

## 3. Component reference

| File | Env | Role |
|---|---|---|
| `train_remote.py` | lerobot | Entry point: loads DP, auto-aligns config, builds VecEnv + SAC, trains. |
| `zmq_client.py` (`ZMQRemoteEnv`) | lerobot | Gymnasium proxy. Spawns one `zmq_server.py` subprocess in the `mimicgen` env, talks to it over `tcp://127.0.0.1:<port>` with pickled `{cmd: reset/step/get_spaces/close}` messages. |
| `zmq_server.py` | mimicgen | Builds `CabinetBowlEnv` (UMI robot, JOINT_POSITION ctrl, 3 cameras), wraps it, serves requests over ZeroMQ. |
| `env_wrapper.py` → `LerobotBaseEnvWrapper` | mimicgen | Per-step obs formatting (cameras → `left/right/wrist`, joints+gripper → `observation.state`, eef → `observation.eef_state`) **and** the absolute-target → JOINT_POSITION-delta action conversion. |
| `env_wrapper.py` → `LerobotActionChunkWrapper` | mimicgen | Expands one `(act_steps × action_dim)` action into `act_steps` sequential env sub-steps; sums reward; enforces `max_episode_steps`. |
| `env_wrapper.py` → `LerobotDiffusionPolicyEnvWrapper` | lerobot | `VecEnvWrapper` for SB3. Turns the SAC noise action into DP initial noise, runs the DP, ships robot actions to the VecEnv, maintains the `n_obs_steps` history queues, exposes the flat state space to SAC. |
| `policy_wrapper.py` → `LerobotDPPOPolicyWrapper` | lerobot | Wraps the LeRobot `DiffusionPolicy` for DSRL: applies the normalization pipeline, overlays the noise window, forces DDIM, calls `generate_actions`. |
| `policy_wrapper.py` → `load_lerobot_policy()` | lerobot | Loads the checkpoint + `make_pre_post_processors` and returns the wrapped policy. |
| `infer_remote.py` | lerobot | Load a trained SAC checkpoint and roll out (single env). |
| `debug_single_env.py`, `debug_normalization.py` | lerobot | Verification tools (see [SETUP.md §6](SETUP.md#6-verify-the-setup-before-full-training)). |

---

## 4. Configuration reference

`train_remote.py` borrows the upstream Hydra config `../dsrl/cfg/robomimic/dsrl_can.yaml` for its *structure*, but **most of that file is ignored** — it loads the LeRobot DP instead of the DPPO base policy the config describes. Know which fields actually do something:

### Fields that are read

| Config field | Default (dsrl_can) | Effect |
|---|---|---|
| `algorithm` | `dsrl_na` → **override to `dsrl_sac`** | selects SAC (noise space) vs. DSRL-NA (unsupported for images) |
| `seed` | 1 | RNG seed for `random`/`numpy`/`torch` |
| `use_wandb` | True | toggles W&B init |
| `wandb.project` | `dsrl` | W&B project name |
| `env.n_envs` | 5 | number of parallel simulator subprocesses (and ports `5555…`) |
| `env.max_episode_steps` | 300 | env sub-steps before truncation (enforced in the chunk wrapper) |
| `train.action_magnitude` | 1.5 | bound of the noise action space: `Box(±mag, size=act_steps·action_dim)` |
| `model.device` | `cuda` (via `cfg.device`) | device the DP and noise tensors live on |
| `render` | false | open one OpenCV GUI window for env 0 |

### Fields that are auto-aligned (do not hand-set)

| Field | Source |
|---|---|
| `action_dim` | `DP.config.output_features["action"].shape[0]` → **8** |
| `act_steps` | `DP.config.n_action_steps` → **8** |

### SAC hyperparameters (hardcoded in `train_remote.py`)

```python
policy_kwargs = dict(net_arch=dict(pi=[2048,2048,2048], qf=[2048,2048,2048]),
                     activation_fn=torch.nn.Tanh, n_critics=2)
SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=1_000_000,
    batch_size=256, tau=0.005, gamma=0.99, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=2_000_000, callback=[CheckpointCallback(save_freq=1000, ...)])
```

> **Deviations from upstream `dsrl/train_dsrl.py` you may want to restore for performance** (see also [Tuning](#8-tuning--known-limitations)): the upstream recipe uses `learning_starts`, a high UTD (`gradient_steps≈20`), `LayerNorm` in the networks, and an `init_rollout_steps` warm-up that pre-fills the replay buffer with base-policy rollouts. `train_remote.py` currently omits these. They are the first things to add if SAC is slow to learn.

---

## 5. Interface contract (correctness-critical)

Because the DP, the simulator, and the RL agent live in separate processes/stacks, the bytes crossing between them must match **exactly** the convention the DP was trained on. These are not stylistic choices — getting any of them wrong silently breaks the base policy (it acts on out-of-distribution inputs and the arm holds still or thrashes, with reward stuck near 0). Each was found and fixed during bring-up; keep them intact.

1. **Normalization happens *outside* `generate_actions()`.** Modern LeRobot keeps normalization in a separate processor pipeline. The wrapper must `preprocessor(batch)` → `generate_actions` → `postprocessor.process_action(actions)`. Loaded via `make_pre_post_processors(policy_cfg, pretrained_path)`. *Skipping this feeds the UNet raw observations and returns actions in normalized space → policy is effectively broken.*

2. **Robot must be `PR_V4P1SR_UMI`** (the UMI-gripper robot), not `Panda`. The UMI gripper is what defines the `gripper0_eye_in_hand` wrist camera the DP saw in training.

3. **Wrist camera is `gripper0_eye_in_hand`**, not `robot0_eye_in_hand`. Camera→key mapping must match the dataset conversion: `topview_custom → observation.images.left`, `up_sideview → observation.images.right`, `gripper0_eye_in_hand → observation.images.wrist`.

4. **Action is an absolute joint target; the controller is delta-only.** The DP outputs an 8-D *absolute* target `[7 joint positions (rad), 1 gripper qpos ∈ [0,0.04]]`. The `JOINT_POSITION` controller (`kp=150`, `output_max=0.08`) expects a normalized *delta* in `[−1,1]`. Convert each sub-step against the freshest measured joints:
   ```python
   joint_ctrl  = clip((target[:7] − cur_joint_pos) / 0.08, −1, 1)   # 0.08 == controller output_max
   gripper_cmd = clip(1.0 − 2.0 * (target[7] / 0.04), −1, 1)         # qpos[0,0.04] → cmd[+1 closed … −1 open]
   ```
   And the **state** sent back to the DP must use the same conventions: joints via `atan2(sin, cos)`, gripper = `mean(robot0_gripper_qpos) · (0.04/0.044)` (physical→training range), eef euler via Robosuite `transform_utils`. Start the controller from `load_controller_config(default_controller="JOINT_POSITION")` then override only `kp`/`output_*` — building the dict from scratch raises `KeyError: 'interpolation'`.

5. **Camera images are vertically flipped.** Robosuite offscreen renders are flipped vertically relative to the saved training videos. `_format_obs` applies `img[::-1]` (equivalent to the online inference client's `flip(rot90(img,2), axis=1)`). Without it the DP sees an upside-down scene and the arm freezes.

6. **The 8-step noise overlays a 16-step UNet horizon.** The DP's UNet horizon is 16 but only `n_action_steps=8` are executed. DSRL perturbs only those 8. The wrapper builds a full `(B,16,8)` Gaussian noise and overwrites the executed window — slice `[n_obs_steps−1 : n_obs_steps−1+n_action_steps] = [1:9]` — with the SAC noise. The rest stays standard Gaussian.

> **Authoritative references** for these conventions are the original deployment scripts (`mimicgen/Inference/cstm_debug_inference.py`, `cstm_mimicgen_online_client.py`) and the data-generation controller config (`JOINT_POSITION`, `kp=150`, `output ±0.08`). If you retrain the DP or change the robot/cameras/controller, re-derive every item above to match.

---

## 6. Algorithm variants: `dsrl_sac` vs `dsrl_na`

| | `dsrl_sac` | `dsrl_na` |
|---|---|---|
| What it is | plain SAC with the action space = the DP's noise space | DSRL **Noise-Aliased**: also distills a Q-function on the original action space |
| Sample efficiency | lower | higher (preferred when usable) |
| Works with **image** policies? | **Yes** | **No** |
| Use here | ✅ **required** | ❌ raises an error |

`dsrl_na` re-denoises actions *during* the optimization step using observations drawn from the replay buffer — but the buffer only stores low-dim states, not images. `LerobotDPPOPolicyWrapper.__call__` raises explicitly if handed a non-dict (image) batch in that path. **For this image-conditioned DP, always use `algorithm=dsrl_sac`.** `dsrl_na` is retained for low-dim policies.

---

## 7. Reward

`CabinetBowlEnv.reward()` provides a **dense, two-phase shaped reward** (this is real shaping — it is *not* sparse):

| Phase | Condition | Reward |
|---|---|---|
| 1 | bowl **not** grasped | `1 − tanh(5·‖gripper − bowl‖)` ∈ [0, 1) — drives the gripper toward the bowl |
| 2 | bowl grasped | `2 + (1 − tanh(5·‖bowl − drawer‖))` ∈ [2, 3) — always > phase 1, so grasping is incentivised |
| bonus | `_check_success()` | `+1` |

`_check_success()` = `drawer_moved AND bowl_in_drawer`. The whole thing is scaled by `reward_scale` (default `1.0`). The per-substep reward is **summed across the 8 sub-steps** by `LerobotActionChunkWrapper` before being returned to SAC.

> Note: `zmq_server.py` passes `reward_shaping=True`, but the current `reward()` computes the dense reward unconditionally — the flag is stored and not used to gate shaping. (Earlier versions of this env returned a sparse binary success reward; the dense version above is what ships now and is what makes SAC trainable.)

---

## 8. Tuning & known limitations

**Key hyperparameters** (per the DSRL paper): `action_magnitude` (bound on the noise action; ~1.5 is a good start) and UTD / `gradient_steps` (~20 for sample efficiency). Large actor/critic MLPs (3×2048, as configured) help. See `dsrl/README.md` for the upstream tuning notes.

**Throughput.** The DP runs DDIM denoising (~100 steps) on *every* env step, so end-to-end throughput is low (~2 fps observed). To speed up, lower the DDIM inference steps (fewer denoising iterations) — at some quality cost — and/or reduce `env.n_envs` if the GPU is the bottleneck (each env still calls the DP).

**Restore the upstream SAC recipe** if learning stalls: add `learning_starts`, a higher UTD, `LayerNorm`, and an `init_rollout_steps` base-policy warm-up to pre-fill the replay buffer (see [§4 deviations](#4-configuration-reference)).

**Visual domain gap.** The live simulator appearance (table color, gripper shading, lighting) can differ slightly from individual training frames. The "varied" DP checkpoint is more robust to this, but if task completion stays poor with a correct interface, suspect the visual gap next.

---

## 9. Inference / evaluation

`infer_remote.py` loads the same DP and a trained SAC checkpoint and rolls out a single env (it imports OpenCV for optional visualization). Point its `model_path` at the same DP checkpoint used for training (keep them in sync — see [SETUP.md §5](SETUP.md#5-paths-you-must-change)) and load your SAC `.zip` checkpoint from `logs/checkpoints/`.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Arm holds still / barely moves; reward ≈ 0 | broken interface contract | run `debug_single_env.py --gaussian`; re-check [§5](#5-interface-contract-correctness-critical), especially normalization (1) and the image flip (5) |
| `KeyError: 'interpolation'` on server start | controller dict built from scratch | start from `load_controller_config(...)` then override only `kp`/`output_*` ([§5.4](#5-interface-contract-correctness-critical)) |
| `make_pre_post_processors` ImportError / `[WARNING] no pre/post-processor` | LeRobot too old | use a LeRobot revision exposing `make_pre_post_processors` ([SETUP.md §2](SETUP.md#2-the-lerobot-environment-orchestrator)) |
| Client hangs after "Starting server with: …" | server crashed at boot, or wrong interpreter | check server stdout; set `MIMICGEN_PYTHON=/opt/miniconda3/envs/mimicgen/bin/python` ([SETUP.md §5](SETUP.md#5-paths-you-must-change)) |
| `Address already in use` on a port | stale server from a previous run | kill leftover `zmq_server.py` processes, or change `env.n_envs` / restart |
| `ModuleNotFoundError: cabinet_bowl_env` / robot `PR_V4P1SR_UMI` not found | wrong Robosuite tree | fix the `sys.path.append` in `zmq_server.py` to your custom Robosuite ([SETUP.md §3](SETUP.md#3-the-mimicgen-environment-simulator)) |
| `DSRL-NA is not supported for image-based policies` | ran with `algorithm=dsrl_na` | use `algorithm=dsrl_sac` ([§6](#6-algorithm-variants-dsrl_sac-vs-dsrl_na)) |
| Very low fps | DDIM denoising cost | lower DDIM inference steps; reduce `env.n_envs` ([§8](#8-tuning--known-limitations)) |
