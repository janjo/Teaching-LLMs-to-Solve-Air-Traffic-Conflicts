from unsloth import FastLanguageModel
from llm_atc.utils import config
from llm_atc.utils.conflict_formatter import ConflictFormatter
from llm_atc.utils.response_formatter import ResponseFormatter
from llm_atc.vae.vae import VAE
from llm_atc.curiosity.curiosity_model import CuriosityModel
from llm_atc.utils.helpers import get_llm_response, parse_llm_response, conflict_within_n_seconds, pad_conflict
from tqdm import tqdm
import pickle
import torch
import os

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.LARGE_MODEL_NAME,
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True
)

vae = VAE()
vae.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../checkpoints/vae_epoch_1950.pt"), map_location="cpu"))
vae.to(config.DEVICE)
vae.eval()

curiosity_model = CuriosityModel()
curiosity_model.to(config.DEVICE)
curiosity_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../checkpoints/curiosity_epoch_29.pt"), map_location="cpu"))

conflict_formatter = ConflictFormatter()
response_formatter = ResponseFormatter()

batch_size = 5000
data = []
scores = []
conflicts = vae.sample_valid(batch_size, device=config.DEVICE, max_attempts=2000)

for conflict in tqdm(conflicts, desc="Generating dataset"):
    acft_list = conflict_formatter.build_aircraft_list(conflict)
    prompt = conflict_formatter.conflict_to_prompt(acft_list)
    llm_output = get_llm_response(model, tokenizer, prompt)
    instr = parse_llm_response(llm_output)
    instr_map = conflict_formatter.map_llm_instructions_to_indices(acft_list, instr)

    action_reward = conflict_within_n_seconds(conflict, instructions_by_index=instr_map, n=90)
    action_array = conflict_formatter.build_action_tensor_from_state(conflict, instr_map)

    state_tensor = torch.tensor(pad_conflict(conflict, node_dim=6)[0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    action_tensor = torch.tensor(pad_conflict(action_array, node_dim=2)[0], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    reward_tensor = torch.tensor(action_reward, dtype=torch.float32).to(config.DEVICE)

    with torch.no_grad():
        curiosity_score = curiosity_model(state_tensor, action_tensor, reward_tensor).item()
        conflict_reward_score = conflict_within_n_seconds(conflict, instr_map, n=90)
        reward = 0.3 * curiosity_score + 0.7 * conflict_reward_score

    data.append((prompt, llm_output, reward))
    scores.append((curiosity_score, conflict_reward_score))

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/conflicts.pkl"), "wb") as f:
    pickle.dump(conflicts, f)

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/conflict_data.pkl"), "wb") as f:
    pickle.dump(data, f)

with open(os.path.join(os.path.dirname(__file__), "../checkpoints/conflict_scores.pkl"), "wb") as f:
    pickle.dump(scores, f)

filtered_data = [entry for entry in data if entry[2] >= 0.7]
response_formatter = ResponseFormatter()
formatted_data = []

for entry in filtered_data:
    response_formatter.load_aircraft_list(conflict_formatter.build_aircraft_list_from_prompt(entry[0]))
    formatted_data.append((entry[0], response_formatter.format_analysis() + "\n\n" + response_formatter.format_instruction(parse_llm_response(entry[1])), entry[2]))
    
with open(os.path.join(os.path.dirname(__file__), "../checkpoints/formatted_conflict_data.pkl"), "wb") as f:
    pickle.dump(formatted_data, f)

