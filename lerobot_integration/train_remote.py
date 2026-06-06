import os
import random
import torch
import numpy as np
import wandb
import hydra
from omegaconf import OmegaConf

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import wrappers
from policy_wrapper import load_lerobot_policy
from env_wrapper import LerobotDiffusionPolicyEnvWrapper
from zmq_client import ZMQRemoteEnv

def make_env_factory(port, act_steps, max_steps, render=False):
    def _init():
        return ZMQRemoteEnv(port, act_steps, max_steps, render=render)
    return _init

@hydra.main(config_path="../dsrl/cfg/robomimic", config_name="dsrl_can.yaml", version_base=None)
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
         wandb.init(project=cfg.wandb.project, name="lerobot_dsrl_cabinet_bowl_zmq", monitor_gym=True)

    # 1. Load LeRobot base policy
    model_path = "/mnt/DataDrive/samaksh/model_weights/DP/dp_100_sim_bowl_pnp_varied_larger_version/pretrained_model"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_policy_wrapper = load_lerobot_policy(model_path, device=dev)

    # Auto-align training configs with checkpoint specifications
    policy_act_dim = base_policy_wrapper.base_policy.config.output_features["action"].shape[0]
    policy_act_steps = getattr(base_policy_wrapper.base_policy.config, "n_action_steps", 8)
    cfg.action_dim = policy_act_dim
    cfg.act_steps = policy_act_steps
    print(f"[INFO] Auto-aligned training config to checkpoint: action_dim={cfg.action_dim}, act_steps={cfg.act_steps}")

    # 2. Setup Vectorized Env via ZeroMQ Subprocesses
    num_env = cfg.env.n_envs
    start_port = 5555

    # Optional on-screen GUI: pass `render=true` on the CLI. Only the first env is
    # rendered (one OpenCV window) to avoid opening one window per parallel env.
    render_gui = bool(OmegaConf.select(cfg, "render", default=False))
    if render_gui:
        print("[INFO] GUI rendering ENABLED for env 0 (pass render=false to disable).")

    envs_list = [
        make_env_factory(start_port + i, cfg.act_steps, cfg.env.max_episode_steps,
                         render=(render_gui and i == 0))
        for i in range(num_env)
    ]
    venv = SubprocVecEnv(envs_list)
    
    # 3. Apply DSRL noise wrapper on top of Vector Env
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
