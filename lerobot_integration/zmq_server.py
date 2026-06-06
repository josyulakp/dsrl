import zmq
import argparse
import pickle
import numpy as np
import sys
import robosuite as suite
from robosuite.environments.base import register_env

# Register custom environment
sys.path.append("/mnt/DataDrive/samaksh/mimicgen/robosuite")
from robosuite.environments.manipulation.cabinet_bowl_env import CabinetBowlEnv
register_env(CabinetBowlEnv)

# Import wrappers locally
from env_wrapper import LerobotBaseEnvWrapper, LerobotActionChunkWrapper

def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--render", action="store_true",
                        help="Open an on-screen OpenCV window (requires a display).")
    args = parser.parse_args()

    # JOINT_POSITION controller matching the dataset generation config
    # (mimicgen/exps/Bowl_PnP.json: kp=150, output ±0.08). Start from the robosuite
    # default (so required fields like 'interpolation' are present) and override only
    # the values the data-gen config set. The absolute-target -> delta conversion in
    # env_wrapper.py divides by this same output_max (0.08).
    from robosuite.controllers import load_controller_config
    controller_config = load_controller_config(default_controller="JOINT_POSITION")
    controller_config["kp"] = 150
    controller_config["output_max"] = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    controller_config["output_min"] = [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08]

    # Robot + wrist camera MUST match the data the DP was trained on:
    # PR_V4P1SR_UMI (UMI gripper) exposes the "gripper0_eye_in_hand" camera.
    env = suite.make(
        "CabinetBowlEnv",
        robots=["PR_V4P1SR_UMI"],
        controller_configs=controller_config,
        has_renderer=args.render,            # on-screen OpenCV window when --render
        render_camera="up_sideview",         # third-person view for the GUI window
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["topview_custom", "up_sideview", "gripper0_eye_in_hand"],
        camera_heights=480,
        camera_widths=640,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap environment
    env = LerobotBaseEnvWrapper(env, render=args.render)
    env = LerobotActionChunkWrapper(env, act_steps=args.act_steps, max_episode_steps=args.max_episode_steps)

    # Setup ZeroMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{args.port}")

    print(f"[ZMQ Server @ {args.port}] Started. Waiting for client to connect...")

    while True:
        try:
            msg = pickle.loads(socket.recv())
        except Exception as e:
            print(f"[ZMQ Server @ {args.port}] Unpickling error: {e}")
            socket.send(pickle.dumps(None, protocol=4))
            continue

        cmd = msg.get('cmd')

        if cmd == 'reset':
            obs = env.reset()
            socket.send(pickle.dumps(obs, protocol=4))
        elif cmd == 'step':
            obs, reward, done, info = env.step(msg['action'])
            # Robosuite returns objects inside info that might be unpicklable if it contains C-structs, 
            # safe fallback: return an empty info dict if it crashes, but standard dicts are fine.
            try:
                socket.send(pickle.dumps((obs, reward, done, info), protocol=4))
            except Exception as e:
                socket.send(pickle.dumps((obs, reward, done, {}), protocol=4))
                
        elif cmd == 'get_spaces':
            obs_spaces = {}
            for k, v in env.observation_space.spaces.items():
                obs_spaces[k] = {
                    'low': v.low.min() if hasattr(v.low, 'min') else v.low,
                    'high': v.high.max() if hasattr(v.high, 'max') else v.high,
                    'shape': v.shape,
                    'dtype': v.dtype
                }
            spaces_info = {
                'act_low': env.action_space.low,
                'act_high': env.action_space.high,
                'act_dtype': env.action_space.dtype,
                'obs_spaces': obs_spaces
            }
            socket.send(pickle.dumps(spaces_info, protocol=4))
            
        elif cmd == 'close':
            env.close()
            socket.send(pickle.dumps("OK", protocol=4))
            print(f"[ZMQ Server @ {args.port}] Shutting down.")
            break

if __name__ == "__main__":
    run_server()
