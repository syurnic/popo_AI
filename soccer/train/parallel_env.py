import torch
import numpy as np
import config
import multiprocessing as mp
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

def _worker(remote, idx):
    env = FlattenObservation(gym.make("SoccerEnv", render_mode="rgb_array"))
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, trunc, info = env.step(data.tolist())
                remote.send((ob, reward, done, trunc, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'render':
                frame = env.render()  # numpy array
                remote.send(frame)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass

class ParallelVecSoccerEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for idx, work_remote in enumerate(self.work_remotes):
            p = mp.Process(target=_worker, args=(work_remote, idx))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.device = "cpu"
        # 관찰/액션 스페이스를 하나의 더미 env로부터 얻음
        dummy_env = FlattenObservation(gym.make("SoccerEnv", render_mode="rgb_array"))
        self.envs = [dummy_env]  # 관찰/액션 스페이스 참조용

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        return obs, infos

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, truncs, infos = zip(*results)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        truncs = torch.tensor(truncs, dtype=torch.bool, device=self.device)
        return obs, rewards, dones, truncs, infos

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        frames = [remote.recv() for remote in self.remotes]
        return frames

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()