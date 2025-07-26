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
        self.n_actions = action_space.n + 2
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

    def _process_action_mapping(self, actions: torch.Tensor) -> torch.Tensor:
        """
        액션 매핑 후처리: 특수 액션 인덱스를 실제 액션으로 변환
        """
        # 1 or 3 (n_actions - 2 인덱스를 1 또는 3으로 랜덤 매핑)
        idx = (actions == self.n_actions - 2)
        if any(idx):
            actions[idx] = torch.randint(0, 2, (idx.sum().item(),), device=actions.device) * 2 + 1
        
        # 2 or 3 (n_actions - 1 인덱스를 2 또는 3으로 랜덤 매핑)
        idx = (actions == self.n_actions - 1)
        if any(idx):
            actions[idx] = torch.randint(0, 2, (idx.sum().item(),), device=actions.device) + 2
        
        return actions

    def select_action(self, state: torch.Tensor, use_cpu: bool = False) -> torch.Tensor:
        """
        state: torch.Tensor, shape [batch, obs_dim]
        return: torch.Tensor, shape [batch, 1]
        """
        batch_size = state.shape[0]
        
        if self.eval_mode:
            # 평가 모드: 낮은 확률로 랜덤 액션 선택
            if random.random() < config.EPS_END * 0.1:
                actions = torch.tensor(
                    [self.action_space.sample() for _ in range(batch_size)], 
                    dtype=torch.long, 
                    device=state.device
                ).reshape((-1, 1))
            else:
                # 그리디 액션 선택
                with torch.no_grad():
                    actions = self.policy_net(state.to(config.device)).max(1).indices.view(batch_size, 1).to(state.device)

            # 액션 매핑 후처리
            return self._process_action_mapping(actions)
        
        # 학습 모드: epsilon-greedy 정책
        sample = torch.rand(batch_size, device=state.device)
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * self.steps_done / config.EPS_DECAY)
        
        self.steps_done += batch_size
        greedy = sample > eps_threshold
        
        # 기본적으로 랜덤 액션으로 초기화
        actions = torch.tensor(
            [self.action_space.sample() for _ in range(batch_size)], 
            dtype=torch.long, 
            device=state.device
        ).reshape((-1, 1))

        # 그리디한 경우 네트워크에서 액션 선택
        with torch.no_grad():
            if use_cpu:
                network_actions = self.target_net_cpu(state[greedy]).max(1).indices.view(-1, 1)
            else:
                network_actions = self.target_net(state[greedy]).max(1).indices.view(-1, 1)
            
            actions[greedy] = network_actions
        
        return actions # 훈련 모드의 경우 후처리를 하지 않음

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