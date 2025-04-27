## run for each variant separately

from unsloth import FastLanguageModel
import torch
from llm_atc.utils.conflict_formatter import ConflictFormatter
from llm_atc.curiosity.curiosity_model import CuriosityModel
from llm_atc.utils.helpers import parse_llm_response, get_llm_response, conflict_within_n_seconds, pad_conflict, set_seed
from llm_atc.utils import config
import pickle
from tqdm import tqdm
import os
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.LARGE_MODEL_NAME,
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True
)

#model = PeftModel.from_pretrained(model, "checkpoints/llm-large")

formatter = ConflictFormatter()

curiosity_model = CuriosityModel()
curiosity_model.to(config.DEVICE)
curiosity_model.load_state_dict(os.path.join(os.path.dirname(__file__), torch.load("checkpoints/curiosity_epoch_29.pt"), map_location="cpu"))

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/conflicts.pkl"), "rb") as f:
    conflicts = pickle.load(f)

test_data = []
test_scores = []

for conflict in tqdm(conflicts, desc="Testing on training data:"):
    acft_list = formatter.build_aircraft_list(conflict)
    prompt = formatter.conflict_to_prompt(acft_list)
    llm_output = get_llm_response(model, tokenizer, prompt)
    instr = parse_llm_response(llm_output)
    instr_map = formatter.map_llm_instructions_to_indices(acft_list, instr)

    action_reward = conflict_within_n_seconds(conflict, instructions_by_index=instr_map, n=90)
    action_array = formatter.build_action_tensor_from_state(conflict, instr_map)

    state_tensor = torch.tensor(pad_conflict(conflict, node_dim=6)[0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    action_tensor = torch.tensor(pad_conflict(action_array, node_dim=2)[0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    reward_tensor = torch.tensor(action_reward, dtype=torch.float32).to(config.DEVICE)

    with torch.no_grad():
        curiosity_score = curiosity_model(state_tensor, action_tensor, reward_tensor).item()
        conflict_reward_score = conflict_within_n_seconds(conflict, instr_map, n=90)
        reward = 0.3 * curiosity_score + 0.7 * conflict_reward_score

    test_data.append((prompt, llm_output, reward))
    test_scores.append((curiosity_score, conflict_reward_score))

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/test_data-large-base.pkl"), "wb") as f:
    pickle.dump(test_data, f)

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/test_scores-large-base.pkl"), "wb") as f:
    pickle.dump(test_scores, f)