import numpy as np
from llm_atc.simulator.aircraft import Aircraft
import random

class ConflictValidator:
    def __init__(self,
                 fl_range=(100, 450),
                 too_late_radius_nm=8.0,
                 predict_conflict_within_sec=90,
                 verbose=True,
                 reject_uniform_fls=True,
                 reject_max_aircraft=True):
        self.fl_range = fl_range
        self.too_late_radius_nm = too_late_radius_nm
        self.predict_conflict_within_sec = predict_conflict_within_sec
        self.verbose = verbose
        self.reject_uniform_fls = reject_uniform_fls
        self.reject_max_aircraft = reject_max_aircraft

    def validate(self, conflict):
        if not self._check_structure(conflict): return False
        aircraft_list = self._build_aircraft(conflict)
        if not self._check_proximity(aircraft_list): return False
        if not self._check_predicted_conflict(aircraft_list): return False
        if not self._check_diversity(conflict): return False

        if self.verbose:
            print("Accepted: Valid conflict detected.")
        return True

    def _check_structure(self, conflict):
        if conflict.shape[0] < 2:
            if self.verbose: print("Rejected: Less than 2 aircraft.")
            return False
        for idx, (x, y, fl, target_fl, hdg) in enumerate(conflict[:, :5]):
            if not np.isfinite([x, y, fl, target_fl, hdg]).all():
                if self.verbose: print(f"Rejected: AC{idx} has non-finite values.")
                return False
            if not (self.fl_range[0] <= fl <= self.fl_range[1]):
                if self.verbose: print(f"Rejected: AC{idx} FL {fl} out of bounds.")
                return False
            if not (self.fl_range[0] <= target_fl <= self.fl_range[1]):
                if self.verbose: print(f"Rejected: AC{idx} target FL {target_fl} out of bounds.")
                return False
        return True

    def _build_aircraft(self, conflict):
        aircraft = []
        for idx, (x, y, fl, target_fl, hdg) in enumerate(conflict[:, :5]):
            aircraft.append(Aircraft(
                identifier=f"AC{idx+1}",
                position=[x, y],
                heading=hdg % 360,
                target_heading=hdg % 360,
                speed=280,
                flight_level=fl,
                target_flight_level=target_fl,
                rate_of_climb=10
            ))
        return aircraft

    def _check_proximity(self, aircraft):
        for i in range(len(aircraft)):
            for j in range(i + 1, len(aircraft)):
                d = np.linalg.norm(aircraft[i].position - aircraft[j].position)
                df = abs(aircraft[i].flight_level - aircraft[j].flight_level)
                if d < self.too_late_radius_nm and df < 10.0:
                    if self.verbose:
                        print(f"Rejected: AC{i+1} and AC{j+1} too close — dist={d:.2f}nm, ΔFL={df:.1f}")
                    return False
        return True

    def _check_predicted_conflict(self, aircraft):
        for t in range(0, self.predict_conflict_within_sec + 1, 5):
            pos = [ac.predict_position(t)[0] for ac in aircraft]
            fls = [ac.predict_position(t)[1] for ac in aircraft]
            for i in range(len(aircraft)):
                for j in range(i + 1, len(aircraft)):
                    d = np.linalg.norm(pos[i] - pos[j])
                    df = abs(fls[i] - fls[j])
                    if d < 5.0 and df < 10.0:
                        return True
        if self.verbose:
            print("Rejected: No predicted conflict in window.")
        return False

    def _check_diversity(self, conflict):
        fls = conflict[:, 2]
        target_fls = conflict[:, 3]

        if self.reject_uniform_fls and np.allclose(fls, fls[0], atol=1.0) and np.allclose(target_fls, target_fls[0], atol=1.0):
            if random.random() < 0.7:
                if self.verbose: print("Rejected: Uniform FLs and target FLs.")
                return False

        presence = conflict[:, -1] if conflict.shape[1] == 6 else np.ones(conflict.shape[0])
        num_present = int(np.sum(presence > 0.5))

        if self.reject_max_aircraft and num_present >= 5 and random.random() < 0.3:
            if self.verbose: print("Rejected: 5-aircraft scenario — rejected to promote diversity.")
            return False
        
        if self.reject_max_aircraft and num_present == 4 and random.random() < 0.2:
            if self.verbose: print("Rejected: 5-aircraft scenario — rejected to promote diversity.")
            return False

        return True