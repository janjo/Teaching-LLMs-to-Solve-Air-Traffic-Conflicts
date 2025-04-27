from .conflict_formatter import ConflictFormatter
from .helpers import aircraft_in_conflict_within_n_seconds

class ResponseFormatter:
    def __init__(self):
        pass

    def load_aircraft_list(self, aircraft_list):
        self.formatter = ConflictFormatter()
        self.aircraft_list = aircraft_list
        self.aircraft_lookup = {ac["identifier"]: ac for ac in aircraft_list}

    def format_instruction(self, instructions):
        lines = ["Instructions:"]
        for callsign, instr in instructions.items():
            current_data = self.aircraft_lookup.get(callsign, {})
            current_fl = current_data.get("level")

            heading = instr.get("heading")
            level = instr.get("level")
            turn_dir = instr.get("turn_direction")

            maintain_heading = instr.get("maintain_heading", False)
            maintain_level = instr.get("maintain_level", False)

            parts = []

            if maintain_heading and maintain_level:
                lines.append(f"{callsign} MAINTAIN PRESENT LEVEL AND HEADING")
                continue

            if maintain_level:
                parts.append("MAINTAIN PRESENT LEVEL")
            elif level is not None:
                if level > current_fl:
                    parts.append(f"CLIMB TO FLIGHT LEVEL {level:03d}")
                elif level < current_fl:
                    parts.append(f"DESCEND TO FLIGHT LEVEL {level:03d}")
                else:
                    parts.append("MAINTAIN PRESENT LEVEL")

            if maintain_heading:
                parts.append("MAINTAIN PRESENT HEADING")
            elif heading is not None:
                if turn_dir in ["LEFT", "RIGHT", "left", "right"]:
                    parts.append(f"TURN {turn_dir.upper()} HEADING {heading:03d} DEGREES")
                else:
                    parts.append(f"FLY HEADING {heading:03d} DEGREES")

            instruction = f"{callsign} " + " AND ".join(parts)
            lines.append(instruction)

        return "\n".join(lines)
    
    def format_analysis(self, n=90):
        conflicts = aircraft_in_conflict_within_n_seconds(self.formatter, self.aircraft_list, {}, time_interval=n)

        if not conflicts:
            return f"Analysis: No aircraft are expected to soon lose separation if no instructions are issued."

        lines = [
            f"Analysis:\nBased on projected trajectories and altitude profiles, the following aircraft are expected to soon lose separation minima if no instructions are issued:"
        ]

        acft_by_id = {ac["identifier"]: ac for ac in self.aircraft_list}

        for ac1_id, ac2_id in conflicts:
            ac1 = acft_by_id[ac1_id]
            ac2 = acft_by_id[ac2_id]

            dx = ac1["x"] - ac2["x"]
            dy = ac1["y"] - ac2["y"]
            horizontal_distance = (dx**2 + dy**2)**0.5
            vertical_sep = abs(ac1["level"] - ac2["level"])

            # Check for level changes that could reduce separation
            level_change = False
            for a, b in [(ac1, ac2), (ac2, ac1)]:
                if a.get("target_fl") is not None and a["level"] != a["target_fl"]:
                    current_vsep = abs(a["level"] - b["level"])
                    future_vsep = abs(a["target_fl"] - b["level"])
                    if future_vsep < current_vsep:
                        level_change = True
                        break

            # Also check for vertical convergence (passing through another's level)
            vertical_convergence = False
            for a, b in [(ac1, ac2), (ac2, ac1)]:
                if a.get("target_fl") is not None:
                    if (
                        a["level"] > b["level"] and a["target_fl"] < b["level"]
                    ) or (
                        a["level"] < b["level"] and a["target_fl"] > b["level"]
                    ):
                        vertical_convergence = True
                        break
                        
            causes = []

            if horizontal_distance < 5.0:
                causes.append("close horizontal distance")

            if vertical_sep < 10.0:
                causes.append("insufficient vertical separation")

            if vertical_convergence:
                causes.append("one aircraft is climbing or descending through the level of the other")
            elif level_change:
                causes.append("change of level reducing vertical separation")

            if not causes:
                causes.append("projected paths intersect despite current separation")

            cause_text = "; ".join(causes)
            lines.append(f"- {ac1_id} and {ac2_id}: {cause_text}.")

        return "\n".join(lines)