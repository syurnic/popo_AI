import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import config
import memory
import performance_log
import threading
import copy

class DQNAgent:
    def __init__(
        self,
        model_cls,
        observation_space,
        action_space,
        label: str,
        memory_capacity: int = 100000,
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
        self.label = label
        
        n_observations = observation_space.shape[0]

        self.policy_net = model_cls(
            n_observations,
            self.n_actions,
            label=label,
            **kwargs
        ).to(config.device)

        self.target_net = model_cls(
            n_observations,
            self.n_actions,
            label=label,
            **kwargs
        ).to(config.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.target_net_cpu = copy.deepcopy(self.target_net).to("cpu")

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=lr,
            amsgrad=True
        )

        self.memory = memory.ReplayMemory(memory_capacity, n_observations)
        self.steps_done = 0
        self.gamma = gamma
        self.tau = tau
        self.eval_mode = False  # 평가 모드 여부
        self.lock_target_net = threading.Lock()
        self.lock_memory = threading.Lock()

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode
        if mode:
            self.policy_net.eval()
        else:
            self.policy_net.train()

    def select_action(self, state: torch.Tensor, use_cpu: bool = False) -> torch.Tensor:
        """
        state: torch.Tensor, shape [batch, obs_dim]
        return: torch.Tensor, shape [batch, 1]
        """
        batch_size = state.shape[0]
        if self.eval_mode:
            if random.random() < config.EPS_END * -1:
                return torch.tensor([self.action_space.sample() for _ in range(batch_size)], dtype=torch.long, device=state.device).reshape((-1, 1))
            else:
                with torch.no_grad():
                    return self.policy_net(state.to(config.device)).max(1).indices.view(batch_size, 1).to(state.device)
        
        sample = torch.rand(batch_size, device=state.device)
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * self.steps_done / config.EPS_DECAY)
        
        self.steps_done += batch_size
        
        greedy = sample > eps_threshold
        
        #actions = torch.zeros((batch_size, 1), dtype=torch.long, device=state.device)
        actions = torch.tensor([self.action_space.sample() for _ in range(batch_size)], dtype=torch.long, device=state.device).reshape((-1, 1))
        with torch.no_grad():
            if use_cpu:
                actions[greedy] = self.target_net_cpu(state[greedy]).max(1).indices.view(-1, 1)
            else:
                actions[greedy] = self.target_net(state[greedy]).max(1).indices.view(-1, 1)
        return actions
        # with torch.no_grad():
        #     if use_cpu:
        #         q_values = self.target_net_cpu(state)
        #     else:
        #         q_values = self.target_net(state)
        #     probs = torch.nn.functional.softmax(q_values, dim=1)
        #     actions = torch.multinomial(probs, num_samples=1)
        #     return actions

    def reset_steps(self):
        """epsilon-greedy step 수를 0으로 초기화"""
        self.steps_done = 0

    def store_transition(self, states, actions, next_states, rewards):
        # states, actions, next_states, rewards: [batch, ...] 형태
        with self.lock_memory:
            self.memory.push_batch(states, actions, next_states, rewards)

    def optimize_model(self):
        with self.lock_target_net:
            if self.memory.size < config.BATCH_SIZE:
                return None
            states, actions, next_states, rewards = self.memory.sample(config.BATCH_SIZE)

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=config.device, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(config.device)

            state_action_values = self.policy_net(states).gather(1, actions)

            next_state_values = torch.zeros(config.BATCH_SIZE, device=config.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            expected_state_action_values = (next_state_values * self.gamma) + rewards.squeeze(1)

            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            return loss.item()

    def soft_update_target(self):
        with self.lock_target_net:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
    
    def update_target_net_cpu(self):
        with self.lock_target_net:
            self.target_net_cpu.load_state_dict(self.target_net.state_dict())
    
    def update_memory_device(self):
        with self.lock_memory:
            self.memory.update_device_tensors(config.device)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=config.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict) 
        self.target_net_cpu.load_state_dict(state_dict)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path) 