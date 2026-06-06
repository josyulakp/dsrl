import numpy as np
import gym
from gym import spaces
import zmq
import subprocess
import pickle
import time
import sys

class ZMQRemoteEnv(gym.Env):
    def __init__(self, port, act_steps, max_episode_steps, render=False):
        # Determine paths logically relative to this script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(script_dir, "zmq_server.py")

        # Start server as a subprocess natively in mimicgen Conda Env.
        # MIMICGEN_PYTHON lets you point at an explicit interpreter (e.g.
        # /opt/miniconda3/envs/mimicgen/bin/python) when `conda activate` is not
        # available in the launching shell. Defaults to the original activate-based cmd.
        mimicgen_python = os.environ.get("MIMICGEN_PYTHON")
        server_args = f"{server_path} --port {port} --act_steps {act_steps} --max_episode_steps {max_episode_steps}"
        if render:
            server_args += " --render"
        if mimicgen_python:
            cmd = f"{mimicgen_python} {server_args}"
        else:
            cmd = f'bash -c "source /opt/miniconda3/bin/activate mimicgen && python {server_args}"'
        print(f"Starting server with: {cmd}")
        self.proc = subprocess.Popen(cmd, shell=True)
        self.port = port
        
        # Connect ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{port}")
        
        # Sleep temporarily depending on how long Robosuite takes to spin up
        time.sleep(8)
        
        # Retrieve space formatting directly from server
        self.socket.send(pickle.dumps({'cmd': 'get_spaces'}, protocol=4))
        spaces_info = pickle.loads(self.socket.recv())
        
        self.action_space = spaces.Box(
            low=spaces_info['act_low'], 
            high=spaces_info['act_high'], 
            dtype=spaces_info['act_dtype']
        )
        
        ob_space = {}
        for k, v in spaces_info['obs_spaces'].items():
            ob_space[k] = spaces.Box(
                low=v['low'], 
                high=v['high'], 
                shape=v['shape'], 
                dtype=v['dtype']
            )
            
        self.observation_space = spaces.Dict(ob_space)

    def reset(self, **kwargs):
        self.socket.send(pickle.dumps({'cmd': 'reset'}, protocol=4))
        return pickle.loads(self.socket.recv())

    def step(self, action):
        self.socket.send(pickle.dumps({'cmd': 'step', 'action': action}, protocol=4))
        obs, reward, done, info = pickle.loads(self.socket.recv())
        return obs, reward, done, info

    def close(self):
        try:
            self.socket.send(pickle.dumps({'cmd': 'close'}, protocol=4))
            self.socket.recv()
        except:
            pass
        self.proc.terminate()
        self.proc.wait()
