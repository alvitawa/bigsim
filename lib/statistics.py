import json
import numpy as np

from typing import Any, Callable
from dataclasses import field

from dataclasses import dataclass
from dataclasses_json import config, DataClassJsonMixin, dataclass_json

from .parameters import Parameters


@dataclass_json
@dataclass
class Statistics:
    """Tracks statistics of a smimulation."""

    iterations: int = 0
    boid_count: Any = field(default_factory=lambda: [])
    school_count: Any = field(default_factory=lambda: [])
    school_sizes: Any = field(default_factory=lambda: [])
    duration: float = 0

    def measure(self, sim):
        self.boid_count.append(int(sim.population.shape[0]))
        clusters = np.unique(sim.labels)
        self.school_count.append(int(clusters.shape[0]))
        school_sizes = np.equal(sim.labels[:, None], clusters[None, :]).sum(axis=0)
        school_sizes.sort()
        self.school_sizes.append(list(int(s) for s in school_sizes[::-1]))

    def to_numpy(self, pars):
        max_measurements = pars.max_steps // pars.resolution + 1

        # boid count array
        boid_count = np.zeros(max_measurements)
        boid_count[: len(self.boid_count)] = self.boid_count

        # school count array
        school_count = np.zeros(max_measurements, dtype=int)
        school_count[: len(self.school_count)] = self.school_count

        # school size matrix
        school_sizes = np.zeros(school_count.shape + (school_count.max(),))
        for i, ss in enumerate(self.school_sizes):
            school_sizes[i, : len(ss)] = ss

        return max_measurements, boid_count, school_count, school_sizes

    def save(self, f):
        with open(f, "w") as file:
            return json.dump(self.to_dict(), file, separators=(",", ":"))

    def load(f):
        with open(f, "r") as file:
            stats = Statistics.from_json(file.read())
            return stats

    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)


def stats_to_numpy(stats: list, pars):
    arrs = list(s.to_numpy(pars) for s in stats)
    s = arrs[0][0]
    parts = tuple(np.array(list(arr[i] for arr in arrs)) for i in range(1, 4))
    return (s,) + parts


def load_logs(path):
    pars = Parameters.load(path + "/pars.json")
    stats = []
    try:
        i = 0
        while True:
            indexstr = str(i)
            stats.append(Statistics.load(f"{path}/stats{indexstr}.json"))
            i += 1
    except FileNotFoundError:
        pass
    return (pars, stats)
