from unsloth import FastLanguageModel
from llm_atc.utils.helpers import get_llm_response, parse_llm_response, conflict_within_n_seconds, pad_conflict
from llm_atc.utils.conflict_formatter import ConflictFormatter
from llm_atc.vae.vae import VAE
from llm_atc.utils import config
from peft import PeftModel
from pathlib import Path
import random
import pickle
import torch
import os

comparisons = [
    {"curiosity": True, "lora": False, "name": config.MODEL_NAME},
    {"curiosity": True, "lora": False, "name": config.LARGE_MODEL_NAME},
    {"curiosity": True, "lora": True, "name": config.MODEL_NAME, "adapter_path": "../checkpoints/llm-7b"},
    {"curiosity": True, "lora": True, "name": config.LARGE_MODEL_NAME, "adapter_path": "../checkpoints/llm-14b"},
    {"curiosity": False, "lora": True, "name": config.LARGE_MODEL_NAME, "adapter_path": "../checkpoints/llm-14b-nc"}
]

results = []

vae = VAE()
vae.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),"../checkpoints/vae_epoch_1950.pt"), map_location="cpu"))
vae.to(config.DEVICE)
vae.eval()

conflict_formatter = ConflictFormatter()

conflicts = vae.sample_valid(300, device=config.DEVICE, max_attempts=100)
conflict_aircraft_list = [conflict_formatter.build_aircraft_list(c) for c in conflicts]

for comparison in comparisons:
    scores = []

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = comparison["name"],
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True
    )

    if comparison["lora"]:
        model = PeftModel.from_pretrained(model, os.path.join(os.path.dirname(__file__), comparison["adapter_path"]))


    for i, acft_list in enumerate(conflict_aircraft_list):
        prompt = conflict_formatter.conflict_to_prompt(acft_list)
        llm_output = get_llm_response(model, tokenizer, prompt)
        instr = parse_llm_response(llm_output)
        instr_map = conflict_formatter.map_llm_instructions_to_indices(acft_list, instr)
        conflict_reward_score = conflict_within_n_seconds(conflicts[i], instr_map, n=90)

        scores.append(conflict_reward_score)
    
    results.append({
        "curiosity": comparison["curiosity"],
        "lora": comparison["lora"],
        "name": comparison["name"],
        "score": sum(scores)/len(scores)
    })

with open(os.path.join(os.path.dirname(__file__),"../checkpoints/testing-results.pkl"), "wb") as f:
    pickle.dump(results, f)
