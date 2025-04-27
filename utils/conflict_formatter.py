import numpy as np
import random
import re

class ConflictFormatter:
    def __init__(self):
        pass

    def generate_callsigns(self, conflict):
        callsigns = set()
        while len(callsigns) < len(conflict):
            callsigns.add(random.randint(100, 999))
        return [f"AC{c}" for c in callsigns]

    def build_aircraft_list(self, conflict_array):
        if not isinstance(conflict_array, np.ndarray):
            conflict_array = np.array(conflict_array)
            
        callsigns = self.generate_callsigns(conflict_array)
        return [
            {
                "identifier": callsigns[i],
                "x": conflict_array[i, 0],
                "y": conflict_array[i, 1],
                "level": conflict_array[i, 2],
                "target_fl": conflict_array[i, 3],
                "heading": conflict_array[i, 4]
            }
            for i in range(len(conflict_array))
        ]
    
    def build_aircraft_list_from_prompt(self, prompt):
        aircraft_list = []

        pattern = re.compile(
            r"(?P<callsign>AC\d+): at position \(X (?P<x>[\d.]+), Y (?P<y>[\d.]+)\), "
            r"flight level (?P<fl>\d+), heading (?P<hdg>\d+), "
            r"(?:(maintaining|descending to|climbing to) flight level (?P<target_fl>\d+))"
        )

        for match in pattern.finditer(prompt):
            groups = match.groupdict()
            aircraft = {
                "identifier": groups["callsign"],
                "x": float(groups["x"]),
                "y": float(groups["y"]),
                "level": int(groups["fl"]),
                "target_fl": int(groups["target_fl"]),
                "heading": int(groups["hdg"])
            }
            aircraft_list.append(aircraft)

        return aircraft_list

    def conflict_to_prompt(self, aircraft_list):
        lines = []
        for ac in aircraft_list:
            current_fl = int(ac["level"])
            target_fl = int(ac["target_fl"])
            verb = "climbing to" if target_fl > current_fl else "descending to" if target_fl < current_fl else "maintaining"
            lines.append(
                f"{ac['identifier']}: at position (X {float(ac['x']):.1f}, Y {float(ac['y']):.1f}), flight level {int(current_fl)}, heading {int(ac['heading'])}, {verb} flight level {target_fl}"
            )
        return "\n".join(lines)
    
    def map_llm_instructions_to_indices(self, aircraft_list, instructions_by_callsign):
        index_mapped = {}
        for idx, ac in enumerate(aircraft_list):
            callsign = ac["identifier"]
            if callsign in instructions_by_callsign:
                instr = instructions_by_callsign[callsign]
                index_mapped[idx] = {}

                if instr.get("maintain_heading", False):
                    index_mapped[idx]["target_heading"] = ac["heading"]
                elif "heading" in instr:
                    index_mapped[idx]["target_heading"] = instr["heading"]

                if instr.get("maintain_level", False):
                    index_mapped[idx]["target_fl"] = ac["level"]
                elif "level" in instr:
                    index_mapped[idx]["target_fl"] = instr["level"]

        return index_mapped


    def build_action_tensor_from_state(self, state_array, instructions_by_index):
        """
        Returns (N, 2) array of [heading, target_fl] per aircraft.
        """
        actions = []
        for i in range(state_array.shape[0]):
            current_heading = state_array[i, 4]
            current_target_fl = state_array[i, 3]

            instr = instructions_by_index.get(i, {})
            heading = instr.get("target_heading", current_heading)
            target_fl = instr.get("target_fl", current_target_fl)
            actions.append([heading, target_fl])

        return np.array(actions, dtype=np.float32)

    def format_conflict_for_llm(self, conflict_array):
        aircraft_list = self.build_aircraft_list(conflict_array)
        prompt = self.conflict_to_prompt(aircraft_list)
        return aircraft_list, prompt
