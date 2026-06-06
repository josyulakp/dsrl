import os
import random
import torch
import numpy as np
import wandb
import hydra
from omegaconf import OmegaConf

import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the custom environment
import sys
sys.path.append("/mnt/DataDrive/samaksh/mimicgen/robosuite")
from robosuite.environments.manipulation.cabinet_bowl_env import CabinetBowlEnv
register_env(CabinetBowlEnv)

# Import wrappers
from policy_wrapper import load_lerobot_policy
from env_wrapper import LerobotBaseEnvWrapper, LerobotActionChunkWrapper, LerobotDiffusionPolicyEnvWrapper

def make_env(cfg):
    env = suite.make(
        "CabinetBowlEnv",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["topview_custom", "up_sideview", "gripper0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        reward_shaping=True,
        control_freq=20,
    )
    env = LerobotBaseEnvWrapper(env)
    env = LerobotActionChunkWrapper(env, act_steps=cfg.act_steps, max_episode_steps=cfg.env.max_episode_steps)
    return env

@hydra.main(config_path="../dsrl/cfg/robomimic", config_name="dsrl_can.yaml", version_base=None)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Overrides for our custom scenario since we are using the base config structure
    if not hasattr(cfg, "act_steps"):
        cfg.act_steps = 8
    if not hasattr(cfg, "action_dim"):
        cfg.action_dim = 7

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
         wandb.init(project=cfg.wandb.project, name="lerobot_dsrl_cabinet_bowl", monitor_gym=True)

    # 1. Load LeRobot base policy
    model_path = "/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_policy_wrapper = load_lerobot_policy(model_path, device=dev)

    # 2. Setup Vectorized Env
    num_env = cfg.env.n_envs
    venv = make_vec_env(lambda: make_env(cfg), n_envs=num_env, vec_env_cls=SubprocVecEnv)
    
    # 3. Apply DSRL noise wrapper
    env = LerobotDiffusionPolicyEnvWrapper(venv, cfg, base_policy_wrapper, n_obs_steps=2)

    # 4. Setup DSRL (SAC-based) algorithm
    policy_kwargs = dict(
        net_arch=dict(pi=[2048, 2048, 2048], qf=[2048, 2048, 2048]),
        activation_fn=torch.nn.Tanh,
        n_critics=2,
    )
    if cfg.algorithm == 'dsrl_sac':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
            verbose=1,
        )
    elif cfg.algorithm == 'dsrl_na':
        model = DSRL(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy_wrapper,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=1,
            tensorboard_log=None,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/checkpoints/")

    # Train
    model.learn(total_timesteps=2000000, callback=[checkpoint_callback])

if __name__ == "__main__":
    main()
