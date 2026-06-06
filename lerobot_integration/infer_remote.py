import os
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import cv2

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import wrappers
from policy_wrapper import load_lerobot_policy
from env_wrapper import LerobotDiffusionPolicyEnvWrapper
from zmq_client import ZMQRemoteEnv

def make_env_factory(port, act_steps, max_steps):
    def _init():
        return ZMQRemoteEnv(port, act_steps, max_steps)
    return _init

@hydra.main(config_path="../dsrl/cfg/robomimic", config_name="dsrl_can.yaml", version_base=None)
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    # 1. Load base policy
    model_path = "/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_policy_wrapper = load_lerobot_policy(model_path, device=dev)

    policy_act_dim = base_policy_wrapper.base_policy.config.output_features["action"].shape[0]
    policy_act_steps = getattr(base_policy_wrapper.base_policy.config, "n_action_steps", 8)
    cfg.action_dim = policy_act_dim
    cfg.act_steps = policy_act_steps
    print(f"[INFO] Auto-aligned eval config to checkpoint: action_dim={cfg.action_dim}, act_steps={cfg.act_steps}")

    # 2. Setup Vectorized Env via ZeroMQ Subprocesses
    # For inference, we typically just need 1 environment
    num_env = 1
    # Use a different starting port to avoid collision if training is still running
    start_port = 6555
    envs_list = [make_env_factory(start_port + i, cfg.act_steps, cfg.env.max_episode_steps) for i in range(num_env)]
    venv = SubprocVecEnv(envs_list)
    
    # 3. Apply DSRL noise wrapper
    env = LerobotDiffusionPolicyEnvWrapper(venv, cfg, base_policy_wrapper, n_obs_steps=2)

    # 4. Load the trained SAC/DSRL algorithm
    # You MUST specify the exact path to one of the checkpoint .zip files
    checkpoint_path = "/mnt/DataDrive/samaksh/DSRL/lerobot_integration/logs/checkpoints/rl_model_20000_steps.zip" # <-- UPDATE THIS ITERATION
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        print("Please modify 'checkpoint_path' in this script manually to point to an existing .zip file inside logs/checkpoints/")
        return

    print(f"[INFO] Loading RL model from {checkpoint_path}...")
    if cfg.algorithm == 'dsrl_sac':
        model = SAC.load(checkpoint_path, env=env)
    elif cfg.algorithm == 'dsrl_na':
        model = DSRL.load(checkpoint_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")

    # 5. Run Inference Episode
    print("[INFO] Starting Evaluation...")
    obs = env.reset()
    dones = [False] * num_env
    total_reward = 0.0
    steps = 0
    
    while not all(dones):
        # Deterministic action prediction for evaluation
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        total_reward += rewards[0]
        steps += 1
        print(f"Step: {steps * cfg.act_steps}, Reward step: {rewards[0]:.3f}, Cumulative: {total_reward:.3f}")

        # Note: If you want to record video, you can extract the images using:
        # dict_obs_batch = env.obs_queues[0][-1]  # Retrieves latest observation dict from the first env
        # img = dict_obs_batch["observation.images.left"]
        # Then use cv2.VideoWriter or torchvision to save it.

    print(f"[INFO] Evaluation Complete. Total Reward: {total_reward:.3f}")

if __name__ == "__main__":
    main()
