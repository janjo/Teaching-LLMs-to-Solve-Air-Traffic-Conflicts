# curiosity_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CuriosityModel(nn.Module):
    def __init__(self, node_dim=6, action_node_dim=2, max_nodes=5):
        super().__init__()
        self.node_dim = node_dim
        self.action_node_dim = action_node_dim
        self.max_nodes = max_nodes

        self.state_encoder = nn.Sequential(
            nn.Linear(node_dim * max_nodes, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_node_dim * max_nodes, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode_state(self, padded_tensor):
        x_flat = padded_tensor.view(padded_tensor.size(0), -1)
        return self.state_encoder(x_flat)

    def encode_action(self, padded_tensor):
        x_flat = padded_tensor.view(padded_tensor.size(0), -1)
        return self.action_encoder(x_flat)

    def forward(self, state_tensor, action_tensor, reward_tensor):
        h_s = self.encode_state(state_tensor)
        h_a = self.encode_action(action_tensor)
        predicted = self.classifier(torch.cat([h_s, h_a], dim=-1))
        return torch.abs(predicted - reward_tensor)

    def score_single(self, state, action, reward, device, pad_fn):
        state_tensor = torch.tensor(
            pad_fn(state, node_dim=6)[0], dtype=torch.float32
        ).unsqueeze(0).to(device)

        action_tensor = torch.tensor(
            pad_fn(action, node_dim=2)[0], dtype=torch.float32
        ).unsqueeze(0).to(device)

        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(device)

        with torch.no_grad():
            surprise = self.forward(state_tensor, action_tensor, reward_tensor)

        return surprise.item()