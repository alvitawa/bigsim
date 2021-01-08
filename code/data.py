from typing import Any, Callable
from dataclasses import field
from boid import Boid

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Gekke optimalizatie
from multiprocessing import Pool, Process, Barrier
from scipy.spatial import distance_matrix

from dataclasses import dataclass

def exponential_weight_function(distances, inner_diameter, outer_diameter):
    pass

def gaussian_pos_wf(distances):
    return stats.norm.pdf(distances/2) - stats.norm.pdf(distances)*2

def gaussian_dir_wf(distances):
    return stats.norm.pdf(distances)

    
def sq_pos_wf(distances):
    close = distances < 0.2
    far = distances < 4 - close
    return -2*close + far

def sq_dir_wf(distances):
    return (distances < 1)

def sq_separate_wf(distances):
    return -(distances < 4).astype(int)

def identity_wf(distances):
    return distances == 0


STRATEGIES = 2

@dataclass
class BoidParameters:
    speed: float = 0.05
    agility: float = 0.95
    weights: Any = (10,1)
    pos_wf: Callable = sq_pos_wf
    dir_wf: Callable = sq_dir_wf

    def __post_init__(self):
        self.weights = np.array(self.weights)
        assert self.weights.shape[-1] == STRATEGIES

@dataclass
class EnvParameters:
    """
        Parameters for the simulation.
    """
    shape: Any=(10, 7)
    boid_count: int = 300
    
    def __post_init__(self):
        self.shape = np.array(self.shape)

    
def generate_population(n, size):
    population = np.random.rand(n, 2, 2)
    population[:, 0, :] *= size

    population[:, 1, :] -= 0.5
    population[:, 1, :] /= np.linalg.norm(population[:, 1, :], axis=1)[:, None]

    return population

class Population:
    """
        This class is for all boids.
    """

    def __init__(self, env_parameters=EnvParameters(), boid_parameters=BoidParameters(), grid_size=(1.0,1.0), box_sight_radius=2):

        
        # Save simulation parameters
        self.env = env_parameters
        self.boid = boid_parameters

        # Algo settings
        self.box_sight_radius = box_sight_radius
        self.grid_size = np.array(grid_size)

        x_boxes = int(np.ceil(self.env.shape[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.env.shape[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(self.env.boid_count, self.env.shape)

    def iterate(self, n=1):
        for _ in range(n):
            grid_coordinates = self.population[:, 0, :] // self.grid_size

            barrier = Barrier(len(self.boxes))

            results = []
            for box in self.boxes:
                idx, new = task(box, self.population, grid_coordinates, self.box_sight_radius, self.boid) # TODO MULTITHREAD MY ASS
                results.append((idx, new))

            for idx, new in results:
                self.population[idx] = new
            
            # wrapping
            self.population[:, 0, 0] %= self.env.shape[0]
            self.population[:, 0, 1] %= self.env.shape[1]

        # with Pool(processes=4) as pool:
        #     results = pool.map(task, parameters)

        
def local_update(inner, outer, pars: BoidParameters):

    # Outer coordinates relative to each inner position
    router = outer[:, None, 0, :] - inner[:, 0, :]

    distances = np.power(router, 2).sum(axis=-1)**0.5


    # Go towards/away from other fish
    pos_weights = pars.pos_wf(distances) # (distances < 1)

    weighed_positions = outer[:, None, 0, :] * pos_weights[:, :, None]
    weighed_means = weighed_positions.sum(axis=0)
    
    ## Separation + Cohesion
    positional_target = weighed_means - inner[:, 0, :]

    # Align direction with other fish
    dir_weights = pars.dir_wf(distances) # (1 < distances) & (distances < 4)
    
    weighed_directions = outer[:, None, 1, :] * dir_weights[:, :, None]
    
    ## Alignment
    directional_target = weighed_directions.sum(axis=0)

    # import pdb; pdb.set_trace()
    
    vectors = [positional_target, directional_target]

    deltas = sum(vectors)
    # deltas = sum(w * v / np.linalg.norm(v, axis=1)[:, None] for v, w in zip(vectors, pars.weights))
    deltas /= np.linalg.norm(deltas, axis=1)[:, None]

    updated_inner = np.copy(inner)
    updated_inner[:, 1, :] += pars.agility * deltas
    updated_inner[:, 1, :] /= np.linalg.norm(updated_inner[:, 1, :], axis=1)[:, None]
    updated_inner[:, 0, :] += pars.speed * updated_inner[:, 1, :]

    return updated_inner

def task(assigned_box, population, grid_coordinates, box_sight_radius, boid_parameters):
    inner_idx = np.all(np.equal(grid_coordinates, assigned_box.T), axis=1)

    outer_idx = np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= box_sight_radius

    new_inner = local_update(population[inner_idx], population[outer_idx], boid_parameters)

    return inner_idx, new_inner
