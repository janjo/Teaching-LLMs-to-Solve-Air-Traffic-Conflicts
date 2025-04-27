from .curiosity_model import CuriosityModel
from llm_atc.utils.helpers import pad_conflict, get_llm_response, parse_llm_response, conflict_within_n_seconds
from llm_atc.utils.conflict_formatter import ConflictFormatter
import numpy as np
import torch
import os

class CuriosityTrainingPipeline:
    def __init__(self, device="cpu"):
        self.formatter = ConflictFormatter()
        self.device = device
        self.model = CuriosityModel().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, state_batch, action_batch, reward_batch):
        self.model.train()

        state_pad = torch.tensor(np.stack([pad_conflict(s, node_dim=6)[0] for s in state_batch]), dtype=torch.float32).to(self.device)
        action_pad = torch.tensor(np.stack([pad_conflict(a, node_dim=2)[0] for a in action_batch]), dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        loss = self.model(state_pad, action_pad, reward_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, synthetic_conflicts, model, tokenizer, epoch_idx=0, batch_size=10):
        state_data, action_data, reward_data = [], [], []

        for c in synthetic_conflicts:
            ac_list = self.formatter.build_aircraft_list(c)
            prompt = self.formatter.conflict_to_prompt(ac_list)
            llm_output = get_llm_response(model, tokenizer, prompt)
            instr = self.formatter.map_llm_instructions_to_indices(ac_list, parse_llm_response(llm_output))

            actions = np.zeros((c.shape[0], 2), dtype=np.float32)
            for i, ac in enumerate(ac_list):
                actions[i, 0] = instr.get(i, {}).get("heading", ac["heading"])
                actions[i, 1] = instr.get(i, {}).get("target_fl", ac["target_fl"])

            reward = conflict_within_n_seconds(c, instr)
            state_data.append(c)
            action_data.append(actions)
            reward_data.append(reward)

        total_loss = 0.0
        for i in range(0, len(state_data), batch_size):
            sb = state_data[i:i + batch_size]
            ab = action_data[i:i + batch_size]
            rb = reward_data[i:i + batch_size]
            total_loss += self.train_step(sb, ab, rb)

        torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__),f"../checkpoints/curiosity_epoch_{epoch_idx}.pt"))

        return total_loss