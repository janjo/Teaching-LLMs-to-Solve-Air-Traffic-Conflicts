import numpy as np
import math

class Aircraft:
    def __init__(self, identifier, position, heading, target_heading, speed, flight_level, target_flight_level, rate_of_climb):
        self.identifier = identifier
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.speed = speed
        self.flight_level = flight_level
        self.rate_of_climb = rate_of_climb
        self.target_heading = target_heading
        self.target_flight_level = target_flight_level

    def _move_and_turn(self, time_interval=20, rate_of_turn=3.0):
        position = self.position.copy()
        heading = self.heading
        flight_level = self.flight_level
        time_step = 1

        for _ in range(int(time_interval / time_step)):
            heading_diff = (self.target_heading - heading + 360) % 360
            if heading_diff > 180:
                heading_diff -= 360

            if abs(heading_diff) > 0:
                turn_amount = min(abs(heading_diff), rate_of_turn * time_step)
                heading += turn_amount * (1 if heading_diff > 0 else -1)
                heading %= 360

            flight_level_diff = self.target_flight_level - flight_level
            if abs(flight_level_diff) > 0:
                climb_amount = min(abs(flight_level_diff), self.rate_of_climb/60 * time_step)
                flight_level += climb_amount * (1 if flight_level_diff > 0 else -1)

            speed_nmi_per_s = self.speed / 3600
            distance = speed_nmi_per_s * time_step
            heading_rad = math.radians(heading)
            dx = distance * math.sin(heading_rad)
            dy = distance * math.cos(heading_rad)
            position += np.array([dx, dy], dtype=np.float32)

        return position, heading, flight_level

    def update_position(self, time_interval=20, rate_of_turn=3.0):
        self.position, self.heading, self.flight_level = self._move_and_turn(
            time_interval, rate_of_turn
        )

    def predict_position(self, time_interval=20, rate_of_turn=3.0):
        position, _, flight_level = self._move_and_turn(
            time_interval, rate_of_turn
        )
        return position, flight_level
