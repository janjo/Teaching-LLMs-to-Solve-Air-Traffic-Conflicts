import torch
import numpy as np
from .conflict_validator import ConflictValidator
from llm_atc.utils.prioritized_replay_buffer import PrioritizedReplayBuffer
from llm_atc.utils.conflict_formatter import ConflictFormatter
from .vae import VAE
from llm_atc.utils.helpers import pad_conflict, conflict_within_n_seconds
import os

def randomly_mask_aircraft(conflict_array, validator, dropout_prob=0.3, decay=0.7, min_acft=2):
    masked = conflict_array.copy()

    if masked.shape[1] == 5:
        masked = np.hstack([masked, np.ones((masked.shape[0], 1), dtype=np.float32)])

    indices = list(range(masked.shape[0]))
    np.random.shuffle(indices)

    for i in indices:
        current_active = np.sum(masked[:, 5] > 0.5)
        if current_active <= min_acft:
            break

        if np.random.rand() < dropout_prob:
            test_masked = masked.copy()
            test_masked[i, 5] = 0.0

            if validator.validate(test_masked):
                masked[i, 5] = 0.0
                dropout_prob *= decay

    return masked

def add_noise(conflict, pos_std=0.5, fl_std=10.0, hdg_std=5.0, target_fl_std=10.0):
    noisy = conflict.copy()
    noisy[:, 0:2] += np.random.normal(0, pos_std, size=(conflict.shape[0], 2))
    noisy[:, 2] += np.random.normal(0, fl_std, size=conflict.shape[0])
    noisy[:, 3] += np.random.normal(0, target_fl_std, size=conflict.shape[0])
    noisy[:, 4] += np.random.normal(0, hdg_std, size=conflict.shape[0])
    noisy[:, 4] %= 360
    return noisy.astype(np.float32)

class VAETrainingPipeline:
    def __init__(self, handcrafted_conflicts, device="cpu", curiosity_model=None, vae=None):
        self.curiosity_model = curiosity_model
        self.device = device
        self.validator = ConflictValidator(verbose=False)
        self.conflict_arrays = [
            np.array([[ac["x"], ac["y"], ac["level"], ac["target_level"], ac["heading"]] for ac in c["aircraft"]], dtype=np.float32)
            for c in handcrafted_conflicts
        ]
        self.vae = vae.to(device) if vae else VAE(latent_dim=32, max_nodes=5).to(device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000)
        self.formatter = ConflictFormatter()

    def train_step(self, conflicts_batch, epoch=0):
        self.vae.train()

        conflicts_batch = [randomly_mask_aircraft(c, self.validator, dropout_prob=min(0.4, 0.2 + epoch * 0.001)) for c in conflicts_batch]
        conflicts_batch = [c for c in conflicts_batch if self.validator.validate(c)]
        if not conflicts_batch:
            return 0.0

        padded, masks = zip(*[pad_conflict(c, node_dim=6) for c in conflicts_batch])
        padded = torch.tensor(np.stack(padded), dtype=torch.float32).to(self.device)
        mask = torch.tensor(np.stack(masks), dtype=torch.bool).to(self.device)

        recon, mu, logvar = self.vae(padded)
        loss = self.vae.compute_loss(recon, padded, mu, logvar, mask, epoch=epoch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, batch_size=8, epoch_idx=0):
        real = [add_noise(c) for c in self.conflict_arrays]

        if epoch_idx < 100 or len(self.replay_buffer.heap) == 0:
            training_data = real
        else:
            mix_ratio = min(0.5, epoch_idx / 200)
            num_synth = int(len(self.conflict_arrays) * mix_ratio)
            synth = self.replay_buffer.sample(num_synth) if self.curiosity_model else self.replay_buffer.sample_random(num_synth)
            training_data = real + synth

        training_data = [c for c in training_data if self.validator.validate(c)]

        total_loss = 0
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            total_loss += self.train_step(batch, epoch=epoch_idx)

        self.vae.eval()
        synthethic = self.vae.sample_valid(200, device=self.device)
        for c in synthethic:
            score = 1.0
            if self.curiosity_model:
                a = np.zeros((c.shape[0], 2), dtype=np.float32)
                ac_list = self.formatter.build_aircraft_list(c)
                instr = {
                    i: {"heading": ac["heading"], "target_fl": ac["target_fl"]}
                    for i, ac in enumerate(ac_list)
                }
                r = conflict_within_n_seconds(c, instr, 90)
                score = self.curiosity_model.score_single(c, a, r, self.device, pad_conflict)
            self.replay_buffer.add(c, score)
        
        if epoch_idx % 50 == 0:
            torch.save(self.vae.state_dict(), os.path.join(os.path.dirname(__file__),f"../checkpoints/vae_epoch_{epoch_idx}.pt"))

        return total_loss
