"""
Lightweight proof: is the diffusion policy in the DSRL wrapper getting VALID inputs?

This does NOT need robosuite. It loads the base policy exactly the way
policy_wrapper.py does, then runs ONE forward pass through:
  (A) the CURRENT DSRL path  -> base_policy.diffusion.generate_actions(raw_batch, noise)
  (B) the CORRECT path       -> preprocessor(raw_batch) -> generate_actions -> postprocessor

and prints the input ranges the network actually sees in each case, plus how
different the resulting actions are.

Run:
    /opt/miniconda3/envs/lerobot/bin/python debug_normalization.py
"""
import torch
import numpy as np

torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
np.set_printoptions(precision=3, suppress=True, linewidth=140)

MODEL = "/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

print("Loading policy + processors ...")
policy = DiffusionPolicy.from_pretrained(MODEL).to(DEV).eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=MODEL)

cfg = policy.config
B = 1
n_obs = cfg.n_obs_steps
horizon = cfg.horizon
act_dim = cfg.output_features["action"].shape[0]
n_act = cfg.n_action_steps
print(f"n_obs_steps={n_obs} horizon={horizon} n_action_steps={n_act} action_dim={act_dim}")
print(f"noise_scheduler_type(config)={cfg.noise_scheduler_type}")

# --- Build a REALISTIC raw observation, exactly the dtype/scale the env wrapper produces ---
# State: robosuite joint angles are roughly in [-2, 2] rad. Use the dataset mean so it's in-distribution.
norm_sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
    f"{MODEL}/policy_preprocessor_step_3_normalizer_processor.safetensors")
state_mean = norm_sd["observation.state.mean"].to(DEV)        # (8,)
eef_mean = norm_sd["observation.eef_state.mean"].to(DEV)      # (6,)

raw = {}
raw["observation.state"] = state_mean.view(1, 1, 8).repeat(B, n_obs, 1).float()
raw["observation.eef_state"] = eef_mean.view(1, 1, 6).repeat(B, n_obs, 1).float()
# Images: env wrapper feeds uint8/255 -> [0,1] floats, shape (B, n_obs, C, H, W)
for k in ["observation.images.left", "observation.images.right", "observation.images.wrist"]:
    img = (torch.rand(B, n_obs, 3, 480, 640, device=DEV) * 0.4 + 0.3)  # plausible [0,1] image
    raw[k] = img

print("\n=== RAW observation that the env wrapper hands to the policy ===")
print(f"  state    range [{raw['observation.state'].min():.3f}, {raw['observation.state'].max():.3f}]  (per-dim joint values)")
print(f"  eef      range [{raw['observation.eef_state'].min():.3f}, {raw['observation.eef_state'].max():.3f}]")
print(f"  images   range [{raw['observation.images.left'].min():.3f}, {raw['observation.images.left'].max():.3f}]  (just /255)")


def stack_images(batch):
    batch = dict(batch)
    batch["observation.images"] = torch.stack(
        [batch[key] for key in cfg.image_features], dim=-4)
    return batch


# Fixed noise so (A) and (B) differ ONLY because of normalization, not RNG.
torch.manual_seed(0)
full_noise = torch.randn(B, horizon, act_dim, device=DEV)

# ---------- (A) CURRENT DSRL path: NO normalization ----------
with torch.no_grad():
    batch_A = stack_images(raw)
    act_A = policy.diffusion.generate_actions(batch_A, noise=full_noise.clone())
print("\n=== (A) CURRENT wrapper path  (generate_actions on RAW obs) ===")
print(f"  state seen by net : [{batch_A['observation.state'].min():.3f}, {batch_A['observation.state'].max():.3f}]")
print(f"  images seen by net: [{batch_A['observation.images'].min():.3f}, {batch_A['observation.images'].max():.3f}]")
print(f"  -> raw action out : [{act_A.min():.3f}, {act_A.max():.3f}]  shape {tuple(act_A.shape)}")
print(f"     these go straight to the robot controller (NOT unnormalized)")

# ---------- (B) CORRECT path: preprocessor -> generate -> postprocessor ----------
with torch.no_grad():
    norm = pre(dict(raw))                 # normalize state(MIN_MAX) + images(MEAN_STD)
    batch_B = stack_images(norm)
    act_B_norm = policy.diffusion.generate_actions(batch_B, noise=full_noise.clone())
    act_B = post.process_action(act_B_norm)   # unnormalize action (MIN_MAX) -> real cmd
print("\n=== (B) CORRECT path  (preprocessor -> generate_actions -> postprocessor) ===")
print(f"  state seen by net : [{batch_B['observation.state'].min():.3f}, {batch_B['observation.state'].max():.3f}]  (MIN_MAX -> ~[-1,1])")
print(f"  images seen by net: [{batch_B['observation.images'].min():.3f}, {batch_B['observation.images'].max():.3f}]  (MEAN_STD)")
print(f"  -> normalized act : [{act_B_norm.min():.3f}, {act_B_norm.max():.3f}]")
print(f"  -> real action out: [{act_B.min():.3f}, {act_B.max():.3f}]  shape {tuple(act_B.shape)}")

# ---------- divergence ----------
a = act_A[:, n_obs - 1:n_obs - 1 + n_act].cpu().numpy().reshape(-1)
b = act_B[:, n_obs - 1:n_obs - 1 + n_act].cpu().numpy().reshape(-1)
print("\n=== HOW WRONG IS THE CURRENT PATH? (executed action window) ===")
print(f"  current (A) first action step: {act_A[0,0].cpu().numpy()}")
print(f"  correct (B) first action step: {act_B[0,0].cpu().numpy()}")
print(f"  L2 difference over action window: {np.linalg.norm(a-b):.3f}")
print(f"  mean |A-B| per dim             : {np.abs(a-b).mean():.3f}")
