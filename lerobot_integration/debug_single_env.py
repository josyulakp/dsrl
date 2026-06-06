"""
Single-env, end-to-end DSRL mechanism debugger.

Spins up ONE robosuite env (fresh port) + the real diffusion policy with the
FIXED normalization pipeline, then drives it for a few steps with random
noise-space actions (no SAC) so you can watch every tensor in the loop:

  RL noise action  --(diffusion policy)-->  robot action  --(env)-->  reward

For each step it prints:
  - RL observation (state handed to SAC) range
  - the noise action range (what SAC would output)
  - normalized obs the diffusion net actually sees (should be ~[-1,1] / MEAN_STD)
  - diffused robot action range (real command space, after un-normalization)
  - reward, done

Run (training must be stopped, or use a free port):
    /opt/miniconda3/envs/lerobot/bin/python debug_single_env.py            # random noise (RL-like)
    /opt/miniconda3/envs/lerobot/bin/python debug_single_env.py --gaussian # DP natural prior (base-policy test)

--gaussian draws the RL noise action from N(0,1) (the diffusion prior) instead of
uniform[-mag,mag], so the diffusion policy runs at its nominal behavior. With the
obs/action interface correct, this should produce purposeful motion and reward > 0.
"""
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

np.set_printoptions(precision=3, suppress=True, linewidth=140)

from policy_wrapper import load_lerobot_policy
from env_wrapper import LerobotDiffusionPolicyEnvWrapper
from zmq_client import ZMQRemoteEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

MODEL = "/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model"
PORT = 7600          # fresh port: avoids 5555 (train) and 6555 (orphan infer)
GAUSSIAN = "--gaussian" in sys.argv
N_STEPS = 38 if GAUSSIAN else 12     # full episode (~38 RL steps * 8 = 304 substeps) for base-policy test
MAX_EP_STEPS = 300
ACTION_MAGNITUDE = 1.5

def _make_env(port, act_steps):
    # top-level factory so forkserver children can pickle/re-create it
    return ZMQRemoteEnv(port, act_steps, MAX_EP_STEPS)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_lerobot_policy(MODEL, device=dev)

    act_dim = base.base_policy.config.output_features["action"].shape[0]
    act_steps = base.base_policy.config.n_action_steps
    print(f"[INFO] action_dim={act_dim} act_steps={act_steps} "
          f"(processors loaded: pre={base.preprocessor is not None}, post={base.postprocessor is not None})")

    cfg = OmegaConf.create({
        "act_steps": act_steps,
        "action_dim": act_dim,
        "train": {"action_magnitude": ACTION_MAGNITUDE},
        "model": {"device": dev},
        "env": {"max_episode_steps": MAX_EP_STEPS},
    })

    # --- instrument the diffusion call so we can see net inputs/outputs ---
    # NOTE: patch the CLASS (not the instance): Python resolves obj(...) via
    # type(obj).__call__, so an instance attribute would be ignored.
    _cls = type(base)
    _orig_call = _cls.__call__
    def _traced_call(self, batch, initial_noise, return_numpy=True):
        raw_state = batch["observation.state"]
        raw_img = batch["observation.images.left"]
        out = _orig_call(self, batch, initial_noise=initial_noise, return_numpy=return_numpy)
        # re-normalize once just to report what the net saw (cheap, no diffusion)
        if self.preprocessor is not None:
            norm = self.preprocessor({k: v for k, v in batch.items()})
            ns, ni = norm["observation.state"], norm["observation.images.left"]
            print(f"    [diffusion] raw_state[{raw_state.min():.2f},{raw_state.max():.2f}] "
                  f"-> net_state[{ns.min():.2f},{ns.max():.2f}] | "
                  f"raw_img[{raw_img.min():.2f},{raw_img.max():.2f}] -> net_img[{ni.min():.2f},{ni.max():.2f}]")
        a = np.asarray(out)
        print(f"    [diffusion] noise_act[{initial_noise.min():.2f},{initial_noise.max():.2f}] "
              f"-> robot_act[{a.min():.2f},{a.max():.2f}]  first_step={a.reshape(-1, act_dim)[0]}")
        return out
    _cls.__call__ = _traced_call

    venv = SubprocVecEnv([lambda: _make_env(PORT, act_steps)])
    env = LerobotDiffusionPolicyEnvWrapper(venv, cfg, base, n_obs_steps=2)

    print("\n[INFO] reset ...")
    obs = env.reset()
    print(f"  RL obs (state to SAC): shape={obs.shape} range[{obs.min():.3f},{obs.max():.3f}]")

    mode = "GAUSSIAN N(0,1) prior (base-policy)" if GAUSSIAN else f"uniform[-{ACTION_MAGNITUDE},{ACTION_MAGNITUDE}] (RL-like)"
    print(f"[INFO] noise mode: {mode}")
    ep_rew = 0.0
    max_rew = 0.0
    for t in range(N_STEPS):
        if GAUSSIAN:
            # DP's natural sampling prior: standard normal noise
            noise_act = np.random.randn(1, act_dim * act_steps).astype(np.float32)
        else:
            # random noise-space action == what an untrained SAC actor emits
            noise_act = np.random.uniform(-ACTION_MAGNITUDE, ACTION_MAGNITUDE,
                                          size=(1, act_dim * act_steps)).astype(np.float32)
        print(f"\nstep {t}: noise action range [{noise_act.min():.2f}, {noise_act.max():.2f}]")
        obs, rew, done, info = env.step(noise_act)
        ep_rew += rew[0]
        max_rew = max(max_rew, rew[0])
        print(f"  -> RL obs[{obs.min():.3f},{obs.max():.3f}]  reward={rew[0]:.4f}  done={done[0]}  ep_rew={ep_rew:.3f}")
        if done[0]:
            print("  [episode ended -> auto-reset]")
            ep_rew = 0.0
    print(f"\n[INFO] max single-step reward seen: {max_rew:.4f}")

    print("\n[INFO] done. Closing env.")
    venv.close()


if __name__ == "__main__":
    main()
