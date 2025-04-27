from .vae_trainer import VAETrainingPipeline
from llm_atc.utils import config
from llm_atc.conflicts import conflicts

vae_pipeline = VAETrainingPipeline(conflicts, device=config.DEVICE)

for epoch in range(2001):
    vae_loss = vae_pipeline.run_epoch(epoch_idx=epoch)
    print(f"Epoch {epoch+1} | VAE Loss: {vae_loss:.2f}")