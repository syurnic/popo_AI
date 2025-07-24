import torch
import random
import config

class ReplayMemory:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)

        self.states_device = self.states.to(config.device)
        self.actions_device = self.actions.to(config.device)
        self.next_states_device = self.next_states.to(config.device)
        self.rewards_device = self.rewards.to(config.device)

    def push(self, state, action, next_state, reward):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.next_states[self.position] = next_state
        self.rewards[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, next_states, rewards):
        batch_size = len(states)
        if batch_size == 0:
            return

        end_pos = self.position + batch_size
        if end_pos <= self.capacity:
            self.states[self.position:end_pos] = states
            self.actions[self.position:end_pos] = actions
            self.next_states[self.position:end_pos] = next_states
            self.rewards[self.position:end_pos] = rewards
        else:
            first_part = self.capacity - self.position
            second_part = batch_size - first_part
            # 앞부분
            self.states[self.position:self.capacity] = states[:first_part]
            self.actions[self.position:self.capacity] = actions[:first_part]
            self.next_states[self.position:self.capacity] = next_states[:first_part]
            self.rewards[self.position:self.capacity] = rewards[:first_part]
            # 뒷부분(0부터)
            self.states[0:second_part] = states[first_part:]
            self.actions[0:second_part] = actions[first_part:]
            self.next_states[0:second_part] = next_states[first_part:]
            self.rewards[0:second_part] = rewards[first_part:]

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=config.device)
        states = self.states_device[idx]
        actions = self.actions_device[idx]
        next_states = self.next_states_device[idx]
        rewards = self.rewards_device[idx]
        return states, actions, next_states, rewards
    
    def update_device_tensors(self, device):
        self.states_device = self.states.to(device)
        self.actions_device = self.actions.to(device)
        self.next_states_device = self.next_states.to(device)
        self.rewards_device = self.rewards.to(device)

    def __len__(self):
        return self.size 