import pickle
from .llm_trainer import LLMTrainer
from llm_atc.utils import config
from llm_atc.utils.helpers import set_seed
from pathlib import Path
import random
import os

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/formatted_conflict_data.pkl"), "rb") as f:
    data = pickle.load(f)

data = [entry for entry in data if entry[2] >= 0.7]
trainer = LLMTrainer(model_name=config.LARGE_MODEL_NAME, device=config.DEVICE)
curriculum_order = [2, 3, 4, 5]
mix_ratio = 0.2

grouped = {}
for entry in data:
    num_ac = entry[0].count("AC")
    if num_ac not in grouped:
        grouped[num_ac] = []
    grouped[num_ac].append(entry)

for i, acft_count in enumerate(curriculum_order):
    if acft_count not in grouped:
        continue

    current_data = grouped[acft_count]
    mixed_data = current_data.copy()

    if i > 0:
        for prev_acft_count in curriculum_order[:i]:
            prev_data = grouped.get(prev_acft_count, [])
            k = int(len(current_data) * mix_ratio)
            mixed_data += random.sample(prev_data, min(k, len(prev_data)))

    print(f"\n=== Training on {acft_count}-aircraft conflicts ({len(current_data)} main + {len(mixed_data) - len(current_data)} mixed) ===")

    dataset = trainer.prepare_dataset(mixed_data, system_prompt=config.CONTROLLER_PROMPT, max_length=1024)
    trainer.train(
        dataset,
        epochs=20,
        batch_size=8,
        model_save_path="../checkpoints/llm-14b-nc",
        log_path="../checkpoints/llm-14b-nc-training-log.pkl"
    )


