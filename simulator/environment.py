from .aircraft import Aircraft
import numpy as np
from llm_atc.utils.config import (
    HORIZONTAL_SEPARATION_MIN,
    VERTICAL_SEPARATION_MIN,
    SEPARATION_VIOLATION_PENALTY,
    NO_VIOLATION_REWARD
)
from llm_atc.utils.helpers import parse_llm_response

class Environment:
    def __init__(self, conflict):
        self.conflict = conflict
        self.aircraft = []
        self.reset()

    def reset(self, conflict=None):
        if conflict is not None:
            self.conflict = conflict

        self.aircraft = [
            Aircraft(
                identifier=ac[0],
                position=[float(ac[1]), float(ac[2])],
                heading=int(ac[5]),
                target_heading=int(ac[5]),
                speed=280,
                flight_level=int(ac[3]),
                target_flight_level=int(ac[4]),
                rate_of_climb=10
            )
            for ac in self.conflict
        ]
        return self.get_textual_description()

    def step(self, llm_response):
        instructions = self.parse_llm_response(llm_response)

        for ac in self.aircraft:
            if ac.identifier in instructions:
                cmd = instructions[ac.identifier]
                if "heading" in cmd:
                    heading_val = cmd["heading"]
                    ac.target_heading = abs(heading_val) % 360
                if "level" in cmd:
                    lvl_val = cmd["level"]
                    ac.target_flight_level = lvl_val

        future_conflicts = self.detect_future_conflicts()

        done = True
        reward = NO_VIOLATION_REWARD

        if future_conflicts:
            reward = SEPARATION_VIOLATION_PENALTY

        for ac in self.aircraft:
            ac.update_position()

        textual_description = self.get_textual_description()

        return textual_description, reward, done, {}
    
    def detect_future_conflicts(self, lookahead_time=120, time_step=1):
        future_conflicts = []
        steps = int(lookahead_time / time_step)

        for i, ac1 in enumerate(self.aircraft):
            for j, ac2 in enumerate(self.aircraft):
                if i >= j:
                    continue

                for step in range(steps):
                    time_elapsed = step * time_step

                    pos1, fl1 = ac1.predict_position(time_elapsed)
                    pos2, fl2 = ac2.predict_position(time_elapsed)

                    horizontal_distance = np.linalg.norm(pos1 - pos2)
                    vertical_separation = abs(fl1 - fl2)

                    horizontal_threshold = HORIZONTAL_SEPARATION_MIN
                    vertical_threshold = VERTICAL_SEPARATION_MIN

                    if horizontal_distance < horizontal_threshold and vertical_separation <= vertical_threshold:
                        future_conflicts.append({
                            "aircraft_1": ac1.identifier,
                            "aircraft_2": ac2.identifier,
                            "horizontal_distance": horizontal_distance,
                            "vertical_separation": vertical_separation,
                            "time_to_conflict": time_elapsed,
                        })
                        break

        return future_conflicts

    def get_textual_description(self):
        descriptions = [
            f"Aircraft {ac.identifier}, "
            f"Coordinates: X {float(ac.position[0]):.1f}, Y {float(ac.position[1]):.1f}, "
            f"Level {int(ac.flight_level)}, "
            f"Heading {int(ac.heading):.1f}°"
            for ac in self.aircraft
        ]
        return "\n".join(descriptions)
    

'''
        future_conflicts = self.detect_future_conflicts()

        if future_conflicts:
            conflict_descriptions = "\n".join(
                f"Potential conflict detected between {conflict['aircraft_1']} and {conflict['aircraft_2']}: "
                f"Predicted Distance = {conflict['horizontal_distance']:.1f} NM, "
                f"Predicted Vertical Separation = {conflict['vertical_separation']:.1f} FL, "
                f"Time to Conflict = {conflict['time_to_conflict']:.1f} seconds."
                for conflict in future_conflicts
            )
            descriptions.append("\nFuture Conflicts:\n" + conflict_descriptions)
        else:
            descriptions.append("\nNo potential conflicts detected.")'
'''