import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import config

STATE_SCALE = np.array([50, 30, 30, 30, 50, 30, 30, 30, 50, 30, 30, 30])
STATE_SCALE = torch.tensor(np.ones_like(STATE_SCALE) / STATE_SCALE, device=config.device).unsqueeze(0)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, label):
        super(DQN, self).__init__()
        d_model = 5
        self.layer1 = nn.Linear(n_observations, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.layer3 = nn.Linear(d_model, n_actions)
        self.label = label

    def forward(self, x):
        x = self.preprocess_state(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) 
    
    def preprocess_state(self, state):
        state = state.clone().detach()
        state *= STATE_SCALE
        if self.label == "Player1":
            state[:, 0:2] -= state[:, 4:6]
            state[:, 8:10] -= state[:, 4:6]
        if self.label == "Player2":
            state[:, 0:2] -= state[:, 8:10]
            state[:, 4:6] -= state[:, 8:10]
        return state