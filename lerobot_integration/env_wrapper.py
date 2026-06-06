import numpy as np
import gym
from gym import spaces
import torch
import cv2
from collections import deque

# --- Canonical action/obs interface constants (must match the DP training/deploy
#     convention and the JOINT_POSITION controller in zmq_server.py). ---
CTRL_OUTPUT_MAX = 0.08    # MUST equal the controller's output_max (data-gen: 0.08)
GRIPPER_MAX_QPOS = 0.04   # DP training-space gripper range [0, 0.04]
GRIPPER_SRC_MAX = 0.044   # UMI physical gripper qpos max

def quat2euler(q):
    """
    Convert a quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)
    """
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
        
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)

class LerobotBaseEnvWrapper(gym.Env):
    """
    Wraps cabinet_bowl_env to extract cameras and format them for Lerobot.
    """
    def __init__(self, env, render=False):
        self.env = env
        self._render = render
        self._render_warned = False
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        else:
            low, high = env.action_spec
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # robosuite is only available in the mimicgen (server) env where this class runs.
        import robosuite.utils.transform_utils as T
        self._T = T
        # Latest measured joint positions (rad), used to convert the DP's absolute
        # joint-position targets into JOINT_POSITION delta commands each step.
        self._cur_joint_pos = np.zeros(7, dtype=np.float32)

        # Wrist camera "gripper0_eye_in_hand" belongs to the UMI gripper (PR_V4P1SR_UMI),
        # matching the DP training data. Mapping topview->left, up_sideview->right matches
        # the dataset conversion script.
        self.camera_names = ["topview_custom", "up_sideview", "gripper0_eye_in_hand"]
        self.lerobot_cam_keys = ["observation.images.left", "observation.images.right", "observation.images.wrist"]
        
        raw_obs = self.env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        ob_space = {}
        for cam, lr_key in zip(self.camera_names, self.lerobot_cam_keys):
            cam_key = f"{cam}_image"
            if cam_key in raw_obs:
                shape = raw_obs[cam_key].shape
                ob_space[lr_key] = spaces.Box(low=0, high=255, shape=(shape[2], shape[0], shape[1]), dtype=np.uint8)
            else:
                ob_space[lr_key] = spaces.Box(low=0, high=255, shape=(3, 256, 256), dtype=np.uint8)
        
        ob_space["observation.state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        ob_space["observation.eef_state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict(ob_space)

    def _format_obs(self, raw_obs):
        obs = {}
        for cam, lr_key in zip(self.camera_names, self.lerobot_cam_keys):
            cam_key = f"{cam}_image"
            if cam_key in raw_obs:
                img = raw_obs[cam_key]
                # robosuite returns offscreen renders vertically flipped relative to how
                # the training videos were saved (verified by frame comparison). Flip the
                # H axis so the DP sees the training orientation. Equivalent to the
                # online inference client's flip(rot90(img,2),axis=1).
                img = img[::-1]
                if len(img.shape) == 3:
                    img = np.transpose(img, (2, 0, 1))
                obs[lr_key] = np.ascontiguousarray(img, dtype=np.uint8)
            else:
                obs[lr_key] = np.zeros(self.observation_space[lr_key].shape, dtype=np.uint8)

        # Joint positions (7) + gripper scalar (1) -> state (8)
        joint_pos = np.arctan2(raw_obs["robot0_joint_pos_sin"], raw_obs["robot0_joint_pos_cos"]).astype(np.float32)
        # Cache fresh joint positions for the absolute->delta action conversion in step().
        self._cur_joint_pos = joint_pos.copy()
        # Gripper: mean of both fingers, physical [0,0.044] -> training [0,0.04] (matches
        # cstm_mimicgen_online_client._extract_state), NOT the raw first-finger qpos.
        grip_scalar = float(np.clip(
            np.mean(raw_obs["robot0_gripper_qpos"].astype(np.float32)) * (GRIPPER_MAX_QPOS / GRIPPER_SRC_MAX),
            0.0, GRIPPER_MAX_QPOS,
        ))
        obs["observation.state"] = np.concatenate([joint_pos, [np.float32(grip_scalar)]])

        # EEF position (3) + EEF rotation Euler (3) -> eef_state (6).
        # Use robosuite transform_utils to match the inference/data convention.
        eef_pos = raw_obs["robot0_eef_pos"].astype(np.float32)
        eef_rot = self._T.mat2euler(self._T.quat2mat(raw_obs["robot0_eef_quat"])).astype(np.float32)
        obs["observation.eef_state"] = np.concatenate([eef_pos, eef_rot])
        return obs

    def reset(self, **kwargs):
        res = self.env.reset()
        raw_obs = res[0] if isinstance(res, tuple) else res
        return self._format_obs(raw_obs)

    def step(self, action):
        # The DP outputs an 8-D ABSOLUTE target: [7 joint positions (rad), 1 gripper qpos in
        # [0,0.04]]. The JOINT_POSITION controller expects a normalized DELTA in [-1,1].
        # Convert against the freshest measured joint positions (cached in _format_obs).
        action = np.asarray(action, dtype=np.float32)
        joint_ctrl = np.clip((action[:7] - self._cur_joint_pos) / CTRL_OUTPUT_MAX, -1.0, 1.0)
        # Gripper: target qpos [0,0.04] -> cmd [-1=open, +1=closed]
        gripper_cmd = float(np.clip(1.0 - 2.0 * (float(action[7]) / GRIPPER_MAX_QPOS), -1.0, 1.0))
        ctrl = np.concatenate([joint_ctrl, [np.float32(gripper_cmd)]]).astype(np.float32)

        res = self.env.step(ctrl)
        if len(res) == 5:
            raw_obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            raw_obs, reward, done, info = res

        if self._render:
            self._safe_render()

        return self._format_obs(raw_obs), reward, done, info

    def _safe_render(self):
        # On-screen OpenCV window; never let a render failure crash training
        # (e.g. headless opencv build or missing display).
        try:
            self.env.render()
        except Exception as e:
            if not self._render_warned:
                print(f"[WARNING] render() failed, disabling GUI: {e}")
                self._render_warned = True
                self._render = False


class LerobotActionChunkWrapper(gym.Env):
    def __init__(self, env, act_steps, max_episode_steps=500):
        super().__init__()
        self.env = env
        self.act_steps = act_steps
        self.action_space = spaces.Box(
            low=np.tile(env.action_space.low, act_steps),
            high=np.tile(env.action_space.high, act_steps),
            dtype=env.action_space.dtype
        )
        self.observation_space = env.observation_space
        self.max_episode_steps = max_episode_steps
        self.ep_step = 0

    def reset(self, **kwargs):
        self.ep_step = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        action_dim = self.env.action_space.shape[0]
        actions = actions.reshape(self.act_steps, action_dim)
        
        reward_sum = 0
        done = False
        info = {}
        obs = None

        for idx in range(self.act_steps):
            obs, reward, done, info = self.env.step(actions[idx])
            reward_sum += reward
            self.ep_step += 1
            if done or self.ep_step >= self.max_episode_steps:
                done = True
                break
                
        return obs, reward_sum, done, info


import gymnasium
try:
    from stable_baselines3.common.vec_env import VecEnvWrapper
    _BaseWrapper = VecEnvWrapper
except ImportError:
    _BaseWrapper = object

class LerobotDiffusionPolicyEnvWrapper(_BaseWrapper):
    """
    VecEnv wrapper for SB3 DSRL. 
    State goes to RL, Dict goes to LeRobot Policy.
    """
    def __init__(self, env, cfg, base_policy, n_obs_steps=2):
        super().__init__(env)
        self.action_horizon = cfg.act_steps
        self.action_dim = cfg.action_dim
        
        # RL uses flat state space
        import gymnasium.spaces as gymnasium_spaces
        state_shape = env.observation_space.spaces["observation.state"].shape
        self.observation_space = gymnasium_spaces.Box(
            low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
        )
        
        # RL action space is noise
        mag = cfg.train.action_magnitude if hasattr(cfg.train, 'action_magnitude') else 1.5
        self.action_space = gymnasium_spaces.Box(
            low=-mag * np.ones(self.action_dim * self.action_horizon),
            high=mag * np.ones(self.action_dim * self.action_horizon),
            dtype=np.float32
        )
        
        self.device = cfg.model.device if hasattr(cfg.model, "device") else "cuda"
        self.base_policy = base_policy
        self.n_obs_steps = n_obs_steps
        self.num_envs = getattr(env, 'num_envs', 1)
        self.obs_queues = [deque(maxlen=n_obs_steps) for _ in range(self.num_envs)]
        
    def _update_queues(self, dict_obs_batch):
        # We need to handle SubprocVecEnv returning stacked dictionaries
        for env_idx in range(self.num_envs):
            single_obs = {k: v[env_idx] for k, v in dict_obs_batch.items()}
            self.obs_queues[env_idx].append(single_obs)
            while len(self.obs_queues[env_idx]) < self.n_obs_steps:
                self.obs_queues[env_idx].append(single_obs)

    def _get_lerobot_batch(self):
        batch = {}
        first_obs = self.obs_queues[0][0]
        
        for k in first_obs.keys():
            env_tensors = []
            for env_idx in range(self.num_envs):
                time_tensors = np.stack([self.obs_queues[env_idx][t][k] for t in range(self.n_obs_steps)])
                env_tensors.append(time_tensors)
                
            stacked = np.stack(env_tensors)
            tens = torch.tensor(stacked, device=self.device)
            if tens.dtype == torch.uint8:
                tens = tens.float() / 255.0
            batch[k] = tens
                
        return batch

    def step_async(self, actions):
        initial_noise = torch.tensor(actions, device=self.device, dtype=torch.float32)
        initial_noise = initial_noise.view(-1, self.action_horizon, self.action_dim)
        
        batch = self._get_lerobot_batch()
        
        diffused_actions = self.base_policy(batch, initial_noise=initial_noise)
        diffused_actions = diffused_actions.reshape(self.num_envs, -1)
        
        self.venv.step_async(diffused_actions)

    def step_wait(self):
        dict_obs_batch, rewards, dones, infos = self.venv.step_wait()
        
        # When done, vec env auto-resets. Info might contain true terminal obs.
        # Align terminal_observation dictionary format with SB3 flat state expectation
        for i in range(len(infos)):
            if "terminal_observation" in infos[i]:
                term_obs = infos[i]["terminal_observation"]
                if isinstance(term_obs, dict) and "observation.state" in term_obs:
                    infos[i]["terminal_observation"] = term_obs["observation.state"]

        self._update_queues(dict_obs_batch)
        rl_obs = dict_obs_batch["observation.state"]
        return rl_obs, rewards, dones, infos

    def reset(self):
        dict_obs_batch = self.venv.reset()
        for q in self.obs_queues:
            q.clear()
        
        self._update_queues(dict_obs_batch)
        rl_obs = dict_obs_batch["observation.state"]
        return rl_obs
