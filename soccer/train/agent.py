import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import config
import memory

class DQNAgent:
    def __init__(
        self,
        model_cls,
        observation_space,
        action_space,
        label: str,
        memory_capacity: int = 10000,
        lr: float = config.LR,
        gamma: float = config.GAMMA,
        tau: float = config.TAU,
        **kwargs
    ):
        """
        DQN 에이전트 초기화
        """
        self.action_space = action_space
        self.n_actions = action_space.n
        n_observations = observation_space.shape[0]
        self.policy_net = model_cls(n_observations, self.n_actions, label=label, **kwargs).to(config.device)
        self.target_net = model_cls(n_observations, self.n_actions, label=label, **kwargs).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = memory.ReplayMemory(memory_capacity)
        self.steps_done = 0
        self.gamma = gamma
        self.tau = tau
        self.eval_mode = False  # 평가 모드 여부

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        if self.eval_mode:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * self.steps_done / config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=config.device, dtype=torch.long)

    def reset_steps(self):
        """epsilon-greedy step 수를 0으로 초기화"""
        self.steps_done = 0

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = memory.Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict) 