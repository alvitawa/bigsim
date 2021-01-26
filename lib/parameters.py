import json
from typing import Any

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
    max_steps: int = 500
    shape: Any = (10, 7)
    boid_count: int = 300
    shark_count: int = 0
    """Number of frames (simulation iterations) between measurements"""
    resolution: int = 40
    cluster_method: str = "LARS_CLUSTERING"

    speed: float = 0.05
    agility: float = 0.2

    speedup_lower_threshold: float = 10
    speedup_upper_threshold: float = 100
    speedup_factor = 2

    separation_weight: float = 1.4
    separation_range: float = 0.2

    cohesion_weight: float = 0.05
    cohesion_range: float = 1.0

    alignment_weight: float = 1.0
    alignment_range: float = 0.8

    obstacle_weight: float = 2
    obstacle_range: float = 0.1

    wall_weight: float = 100
    wall_range: float = 0.1

    shark_weight: float = 1.8
    shark_range: float = 0.7

    shark_cohesion_range: float = 1.0
    shark_cohesion_weight: float = 0.05

    shark_separation_range: float = 1
    shark_separation_weight: float = 15

    shark_speed: float = 0.05
    shark_agility: float = 0.09
    shark_wonder_speed: float = 0.045
    shark_charge_speed: float = 0.055
    shark_eaten_speed: float = 0.015
    shark_chase_range: float = 0.5
    shark_chase_duration: float = 30
    shark_cooldown_duration: float = 20
    shark_top_zoveel: float = 10

    shark_eat_range: float = 0.2
    sharks_eat_single: bool = True

    def load(f):
        with open(f, "r") as file:
            pars = Parameters.from_json(file.read())
            return pars

    def save(self, f):
        with open(f, "w") as file:
            return json.dump(self.to_dict(), file, indent=4, sort_keys=True)

    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)
