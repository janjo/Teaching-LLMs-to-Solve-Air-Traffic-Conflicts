import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .conflict_validator import ConflictValidator

class VAE(nn.Module):
    def __init__(self, latent_dim=32, max_nodes=5):
        super(VAE, self).__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.decoder_output_dim = 6  # [x, y, fl, target_fl, heading, presence]
        input_dim = self.decoder_output_dim
        self.validator = ConflictValidator(verbose=False)

        # ------- Encoder -------
        self.encoder = nn.Sequential(
            nn.Linear(max_nodes * input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, max_nodes * self.decoder_output_dim)
        )

    def encode(self, padded_batch):
        x_flat = padded_batch.view(padded_batch.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        out = out.view(-1, self.max_nodes, self.decoder_output_dim)

        # Decode + constrain
        x = (out[:, :, 0] + torch.randn_like(out[:, :, 0]) * 1.0).clamp(8, 36)#.clamp(0, 50)
        y = (out[:, :, 1] + torch.randn_like(out[:, :, 1]) * 1.0).clamp(8, 36)#.clamp(0, 50)
        fl = (F.relu(out[:, :, 2]) + 100 + torch.randn_like(out[:, :, 2]) * 5).clamp(100, 450)
        target_fl = (F.relu(out[:, :, 3]) + 100 + torch.randn_like(out[:, :, 3]) * 5).clamp(100, 450)
        hdg = out[:, :, 4] % 360
        presence = torch.sigmoid(out[:, :, 5] + torch.randn_like(out[:, :, 5]) * 0.1)
        ## MAKE PRESENCE CHECK BE HIGHER, TAKE A NUMBER FROM DISTRIBUTION 0-1. IF ABOVE 0.2, IGNORE AIRCRAFT AND SET PRESENCE=0
        ## EXAMPLE: IF >= 0.4 THEN SET PRESENCE=0 ELSE PRESENCE=1-(GENERATED_NUMBER) || OR 1 - GENERATED_NUMBER
        return torch.stack([x, y, fl, target_fl, hdg, presence], dim=-1)

    def forward(self, padded_batch):
        mu, logvar = self.encode(padded_batch)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar, mask, epoch=None):
        recon_loss = F.mse_loss(recon_x[mask], x[mask])
        kl_weight = max(0.01, min(1.0, epoch / 100)) if epoch is not None else 1.0
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if epoch is not None:
            print(f"[VAE] Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | Weight: {kl_weight:.2f}")

        return recon_loss + kl_weight * kl_loss

    def sample_valid(self, target_count, batch_size=32, device="cpu", max_attempts=20):
        valid_conflicts = []
        attempts = 0

        while len(valid_conflicts) < target_count and attempts < max_attempts:
            z = torch.randn((batch_size, self.latent_dim)).to(device)
            samples = self.decode(z).detach().cpu().numpy()

            for sample in samples:
                present = sample[:, -1] > 0.5
                active_aircraft = sample[present][:, :5]  # [x, y, fl, target_fl, hdg]

                if active_aircraft.shape[0] == 0:
                    continue  # skip if no active aircraft

                active_aircraft[:, 0] = np.round(active_aircraft[:, 0], 1)  # x
                active_aircraft[:, 1] = np.round(active_aircraft[:, 1], 1)  # y
                active_aircraft[:, 2] = (np.round(active_aircraft[:, 2] / 10) * 10).astype(int)  # fl
                active_aircraft[:, 3] = (np.round(active_aircraft[:, 3] / 10) * 10).astype(int)  # target_fl
                active_aircraft[:, 4] = np.round(active_aircraft[:, 4]).astype(int)  # hdg

                n_acft = active_aircraft.shape[0]
                if n_acft <= 4:
                    for j in range(n_acft):
                        offset_x = np.round(np.random.uniform(0, 3), 1)
                        offset_y = np.round(np.random.uniform(0, 3), 1)

                        if active_aircraft[j, 0] <= 10.0:
                            active_aircraft[j, 0] += offset_x
                        elif active_aircraft[j, 0] >= 34.0:
                            active_aircraft[j, 0] -= offset_x

                        if active_aircraft[j, 1] <= 10.0:
                            active_aircraft[j, 1] += offset_y
                        elif active_aircraft[j, 1] >= 34.0:
                            active_aircraft[j, 1] -= offset_y

                if self.validator.validate(active_aircraft):
                    valid_conflicts.append(active_aircraft)
                    if len(valid_conflicts) >= target_count:
                        break

            attempts += 1

        if len(valid_conflicts) == 0:
            print("[VAE] Warning: No valid conflicts found.")
        else:
            print(f"[VAE] Collected {len(valid_conflicts)} valid conflicts after {attempts} attempts.")

        return valid_conflicts[:target_count]