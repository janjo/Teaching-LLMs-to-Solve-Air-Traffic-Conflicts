import numpy as np
import random
import json
import re
from .config import DEVICE, CONTROLLER_PROMPT
import torch
from llm_atc.simulator.aircraft import Aircraft
from transformers import pipeline

import re

def clean_response(aircraft_list, full_text):
    split_text = full_text.strip().split("Instructions:", 1)
    if len(split_text) != 2:
        return full_text

    before, instructions_part = split_text
    instruction_lines = instructions_part.strip().splitlines()

    acft_lookup = {ac["identifier"]: ac for ac in aircraft_list}
    cleaned_lines = []

    for line in instruction_lines:
        match = re.match(r'(AC\d+)\s+(.*)', line.strip())
        if not match:
            continue

        ac_id, command = match.groups()
        ac = acft_lookup.get(ac_id)
        if not ac:
            continue

        keep = True
        command_upper = command.upper()

        if "MAINTAIN PRESENT HEADING" in command_upper:
            if ac["heading"] == ac.get("target_heading", ac["heading"]):
                keep = False

        if "MAINTAIN PRESENT LEVEL" in command_upper:
            if ac["level"] == ac.get("target_fl", ac["level"]):
                keep = False

        if "MAINTAIN PRESENT LEVEL AND HEADING" in command_upper:
            if ac["level"] == ac.get("target_fl", ac["level"]) and ac["heading"] == ac.get("target_heading", ac["heading"]):
                keep = False

        level_match = re.search(r'(?:CLIMB|DESCEND) TO (?:FLIGHT LEVEL|FL)\s*(\d+)', command_upper)
        if level_match:
            target_level = int(level_match.group(1))
            if ac["level"] == target_level or ac.get("target_fl") == target_level:
                keep = False

        if keep:
            cleaned_lines.append(line.strip())

    cleaned_instructions = "\n".join(cleaned_lines)
    return f"{before.strip()}\n\nInstructions:\n{cleaned_instructions}"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_llm_response(model, tokenizer, state_text):
    messages = [
        {"role": "system", "content": CONTROLLER_PROMPT},
        {"role": "user", "content": state_text}
    ] 

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    tokenized_input = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    prompt_len = tokenized_input.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(**tokenized_input, max_new_tokens=256, temperature=0.5)

    new_tokens = output[0][prompt_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response_text

def parse_llm_response(llm_response):
    instructions = {}
    llm_response = re.sub(r'(?:[:\-°]|Aircraft|AIRCRAFT|ACFT|Acft)', '', llm_response)
    issued_instructions = llm_response.split("\n")

    pattern = re.compile(
        r'\b('
        r'MAINTAIN PRESENT LEVEL AND HEADING|'
        r'MAINTAIN LEVEL AND HEADING|'
        r'MAINTAIN PRESENT HEADING AND LEVEL|'
        r'MAINTAIN HEADING AND LEVEL|'
        r'MAINTAIN PRESENT HEADING|'
        r'MAINTAIN CURRENT HEADING|'
        r'MAINTAIN HEADING|'
        r'MAINTAIN PRESENT LEVEL|'
        r'MAINTAIN CURRENT LEVEL|'
        r'MAINTAIN LEVEL|'
        r'TURN LEFT HEADING|'
        r'TURN RIGHT HEADING|'
        r'DESCEND TO|'
        r'CLIMB TO|'
        r'FLY HEADING'
        r')\b'
        r'(?:\s+(?:FLIGHT LEVEL\s+|FL))?'
        r'\s*(\d+(?:\.\d+)?)?\b',
        re.IGNORECASE
    )

    for instruction in issued_instructions:
        tokens = instruction.strip().split()
        if not tokens or len(tokens) > 25:
            continue
        callsign = tokens[0]
        matches = pattern.findall(instruction.strip())
        if matches:
            if callsign not in instructions:
                instructions[callsign] = {}
            for command, value in matches:
                cmd = command.upper()
                if cmd in {"DESCEND TO", "CLIMB TO"} and value:
                    instructions[callsign]["level"] = int(float(value))
                if cmd in {"FLY HEADING", "TURN LEFT HEADING", "TURN RIGHT HEADING"} and value:
                    instructions[callsign]["heading"] = int(float(value))
                    instructions[callsign]["turn_direction"] = (
                        "left" if "LEFT" in cmd else "right" if "RIGHT" in cmd else None
                    )
                if cmd in {"MAINTAIN PRESENT HEADING", "MAINTAIN CURRENT HEADING", "MAINTAIN HEADING"}:
                    instructions[callsign]["maintain_heading"] = True
                if cmd in {"MAINTAIN PRESENT LEVEL", "MAINTAIN CURRENT LEVEL", "MAINTAIN LEVEL"}:
                    instructions[callsign]["maintain_level"] = True
                if cmd in {
                    "MAINTAIN PRESENT LEVEL AND HEADING",
                    "MAINTAIN LEVEL AND HEADING",
                    "MAINTAIN PRESENT HEADING AND LEVEL",
                    "MAINTAIN HEADING AND LEVEL"
                }:
                    instructions[callsign]["maintain_heading"] = True
                    instructions[callsign]["maintain_level"] = True

    return instructions


def predict_next_state(conflict_array, instructions_by_index=None, time_interval=60):
    aircraft_list = []
    for idx, row in enumerate(conflict_array):
        x, y, level, target_level, heading = row
        instr = instructions_by_index.get(idx, {}) if instructions_by_index else {}

        if instr.get("maintain_level"):
            final_target_fl = level
        else:
            final_target_fl = instr.get("target_fl", target_level)
            if final_target_fl is None:
                final_target_fl = level

        if instr.get("maintain_heading"):
            final_target_heading = heading
        else:
            final_target_heading = instr.get("heading", heading)

        aircraft = Aircraft(
            identifier=str(idx),
            position=[x, y],
            heading=heading,
            target_heading=final_target_heading,
            speed=280,
            flight_level=level,
            target_flight_level=final_target_fl,
            rate_of_climb=10
        )

        aircraft.update_position(time_interval=time_interval)
        aircraft_list.append(aircraft)

    next_state_array = np.array([
        [ac.position[0], ac.position[1], ac.flight_level, ac.target_flight_level, ac.heading]
        for ac in aircraft_list
    ], dtype=np.float32)

    return next_state_array

def conflict_within_n_seconds(conflict_array, instructions_by_index=None, n=60):
    for t in range(1, n+1):
        state = predict_next_state(conflict_array=conflict_array, instructions_by_index=instructions_by_index, time_interval=t)
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                dx = state[i][0] - state[j][0]
                dy = state[i][1] - state[j][1]
                d = np.sqrt(dx**2 + dy**2)
                df = abs(state[i][2] - state[j][2])
                if d < 5.0 and df < 10.0:
                    return 0.0
    return 1.0

def aircraft_in_conflict_within_n_seconds(formatter, ac_list, instructions={}, time_interval=60):
    conflict_array = np.array([
        [ac["x"], ac["y"], ac["level"], ac["target_fl"], ac["heading"]] for ac in ac_list
    ], dtype=np.float32)

    instr_by_index = formatter.map_llm_instructions_to_indices(ac_list, instructions)

    callsign_pairs = []
    for t in range(1, time_interval + 1):
        state = predict_next_state(conflict_array, instructions_by_index=instr_by_index, time_interval=t)
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                dx = state[i][0] - state[j][0]
                dy = state[i][1] - state[j][1]
                d = np.sqrt(dx**2 + dy**2)
                df = abs(state[i][2] - state[j][2])
                if d < 5.0 and df < 10.0:
                    pair = tuple(sorted([ac_list[i]["identifier"], ac_list[j]["identifier"]]))
                    if pair not in callsign_pairs:
                        callsign_pairs.append(pair)
    return callsign_pairs

def pad_conflict(conflict_array, max_nodes=5, node_dim=6):
    padded = np.zeros((max_nodes, node_dim), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)

    n = min(conflict_array.shape[0], max_nodes)

    if conflict_array.shape[1] == 5:
        conflict_array = np.hstack([conflict_array, np.ones((conflict_array.shape[0], 1), dtype=np.float32)])

    padded[:n] = conflict_array[:n]
    mask[:n] = 1.0
    return padded, mask
