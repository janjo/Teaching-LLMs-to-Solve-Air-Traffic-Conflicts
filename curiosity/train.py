from unsloth import FastLanguageModel
from .curiosity_trainer import CuriosityTrainingPipeline
from llm_atc.utils import config
from llm_atc.vae.vae import VAE
import torch
import os

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.LARGE_MODEL_NAME,
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True
)

vae = VAE()
vae.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),"../checkpoints/vae_epoch_1950.pt"), map_location="cpu"))
vae.to(config.DEVICE)
vae.eval()

curiosity_pipeline = CuriosityTrainingPipeline(device=config.DEVICE)

for epoch in range(31):
    training_samples = vae.sample_valid(100, device=config.DEVICE, max_attempts=50)
    curiosity_loss = curiosity_pipeline.run_epoch(training_samples, model, tokenizer, epoch)
    print(f"Epoch {epoch+1} | Curiosity Loss: {curiosity_loss:.2f}")