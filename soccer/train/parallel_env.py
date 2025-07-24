import torch
import numpy as np
import config
from concurrent.futures import ThreadPoolExecutor

class ParallelVecSoccerEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        self.device = "cpu"

    def reset(self):
        obs, infos = [], []
        for env in self.envs:
            o, i = env.reset()
            obs.append(o)
            infos.append(i)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        return obs, infos

    def step(self, actions):
        # actions: Tensor of shape [num_envs, 2]
        obs, rewards, dones, truncs, infos = [], [], [], [], []

        def step_env(args):
            env, action = args
            return env.step(action.tolist())

        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            results = list(executor.map(step_env, zip(self.envs, actions)))

        for o, r, d, t, info in results:
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            truncs.append(t)
            infos.append(info)

        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        truncs = torch.tensor(truncs, dtype=torch.bool, device=self.device)
        return obs, rewards, dones, truncs, infos

    def render(self):
        # 각 환경의 render 결과를 리스트로 반환
        return [env.render() for env in self.envs]