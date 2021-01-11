from typing import Any, Callable
from dataclasses import field
from boid import Boid

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Gekke optimalizatie
from multiprocessing import Pool#, Process, Barrier
from scipy.spatial import distance_matrix
from functools import partial

from dataclasses import dataclass

def exponential_weight_function(distances, inner_diameter, outer_diameter):
    pass

def gausst(distances, range, weight):
    result = stats.norm.pdf(distances / range) * np.exp(weight)
    return result

def gaussian_pos_wf(distances, pars):
    cohesion = stats.norm.pdf(distances/pars.cohesion_range)*np.exp(pars.cohesion_weight)
    separation = stats.norm.pdf(distances/pars.separation_range)*np.exp(pars.separation_weight)
    return cohesion - separation

def gaussian_dir_wf(distances, pars):
    return stats.norm.pdf(distances/pars.alignment_range)*np.exp(pars.alignment_weight)

def gaussian_obs_wf(distances, pars):
    return - stats.norm.pdf(distances/pars.obstacle_range) * np.exp(pars.obstacle_weight)
    
def sq_pos_wf(distances, pars):
    close = distances < pars.separation_range
    far = distances < pars.cohesion_range - close
    return -pars.separation_weight*close + pars.cohesion_weight*far

def sq_dir_wf(distances, pars):
    return (distances < pars.alignment_range)*pars.alignment_weight

def identity_wf(distances, _=None):
    return distances == 0

@dataclass
class Parameters:
    shape: Any=(10, 7)
    boid_count: int = 300

    speed: float = 0.04
    agility: float = 0.1

    separation_weight: float = 4
    separation_range: float = 0.1

    cohesion_weight: float = 1
    cohesion_range: float = 0.4

    alignment_weight: float = 0.5
    alignment_range: float = 0.4

    obstacle_weight: float = 100
    obstacle_range: float = 0.7

    pos_wf: Callable = gaussian_pos_wf
    dir_wf: Callable = gaussian_dir_wf
    obs_wf: Callable = gaussian_obs_wf

    def position_weights(self, distances):
        return self.pos_wf(distances, self)

    def direction_weights(self, distances):
        return self.dir_wf(distances, self)

    def position_weights(self, distances):
        return self.pos_wf(distances, self)

    def obstacle_weights(self, distances):
        return self.obs_wf(distances, self)

    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)

    def __post_init__(self):
        self.shape = np.array(self.shape)

    
def generate_population(n, env_size):
    population = np.random.rand(n, 2, 2)
    population[:, 0, :] *= env_size

    population[:, 1, :] -= 0.5
    population[:, 1, :] /= np.linalg.norm(population[:, 1, :], axis=1)[:, None]

    return population

def generate_obstacles(n, env_size):
    obstacles = np.random.rand(n, 1, 2)
    obstacles[:, 0, :] *= env_size

    return obstacles



class Population:
    """
        This class is for all boids.
    """

    def __init__(self, pars=Parameters(), grid_size=(1.0,1.0), box_sight_radius=2):

        
        # Save simulation parameters
        self.pars = pars

        # Algo settings
        self.box_sight_radius = box_sight_radius
        self.grid_size = np.array(grid_size)

        x_boxes = int(np.ceil(self.pars.shape[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.pars.shape[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(self.pars.boid_count, self.pars.shape)

        # make obstacles
        self.obstacles = generate_obstacles(5, self.pars.shape)

    def iterate(self, pool, n=1):
        for _ in range(n):
            grid_coordinates = self.population[:, 0, :] // self.grid_size

            # barrier = Barrier(len(self.boxes))

            # results = []
            # for box in self.boxes:
            #     idx, new = task(box, self.population, grid_coordinates, self.box_sight_radius, self.pars) # TODO MULTITHREAD MY ASS
            #     results.append((idx, new))
            
            
            results = pool.map(partial(task,
                                        population=self.population, 
                                        grid_coordinates=grid_coordinates, 
                                        box_sight_radius=self.box_sight_radius, 
                                        pars=self.pars, obstacles=self.obstacles), 
                                self.boxes)

            for idx, new in results:
                self.population[idx] = new
            
            # wrapping
            self.population[:, 0, 0] %= self.pars.shape[0]
            self.population[:, 0, 1] %= self.pars.shape[1]

            # solid walls
            # self.population[:, 0, 0] = np.clip(self.population[:, 0, 0], 0, self.pars.shape[0])
            # self.population[:, 0, 1] = np.clip(self.population[:, 0, 1], 0, self.pars.shape[1])

        # with Pool(processes=4) as pool:
        #     results = pool.map(task, parameters)

        
def local_update(inner, outer, pars: Parameters, obstacles):

    # Outer coordinates relative to each inner position
    router = outer[:, None, 0, :] - inner[:, 0, :]

    distances = np.power(router, 2).sum(axis=-1)**0.5

    # Go towards/away from other fish
    pos_weights = pars.position_weights(distances) # (distances < 1)

    weighed_positions = router * pos_weights[:, :, None]

    ## Separation + Cohesion
    positional_target = weighed_positions.sum(axis=0)


    # Align direction with other fish
    dir_weights = pars.direction_weights(distances) # (1 < distances) & (distances < 4)
    
    weighed_directions = outer[:, None, 1, :] * dir_weights[:, :, None]
    
    ## Alignment
    directional_target = weighed_directions.sum(axis=0)

    # import pdb; pdb.set_trace()

    # --- OBSTACLES ---
    rel_obstacles = obstacles - inner[:, 0, :]

    distances = np.power(rel_obstacles, 2).sum(axis=-1)**0.5

    # Go away from stuff
    obs_weights = pars.obstacle_weights(distances) # (distances < 1)

    weighed_positions = rel_obstacles * obs_weights[:, :, None]

    ## Avoid points, sharks and walls
    obstacle_target = weighed_positions.sum(axis=0)


    # --- COMBINE --
    
    vectors = [positional_target, directional_target, obstacle_target]

    deltas = sum(vectors)
    # deltas = sum(w * v / np.linalg.norm(v, axis=1)[:, None] for v, w in zip(vectors, pars.weights))
    deltas /= np.linalg.norm(deltas, axis=1)[:, None]

    updated_inner = np.copy(inner)
    updated_inner[:, 1, :] += pars.agility * deltas
    updated_inner[:, 1, :] /= np.linalg.norm(updated_inner[:, 1, :], axis=1)[:, None]
    updated_inner[:, 0, :] += pars.speed * updated_inner[:, 1, :]

    nans = np.argwhere(np.isnan(updated_inner))
    if nans.shape[0] > 0:
        raise Exception(f"{nans.shape[0]} NaN's encountered in local_update")

    return updated_inner

def task(assigned_box, population, grid_coordinates, box_sight_radius, pars, obstacles):
    inner_idx = np.all(np.equal(grid_coordinates, assigned_box.T), axis=1)

    outer_idx = np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= box_sight_radius

    new_inner = local_update(population[inner_idx], population[outer_idx], pars, obstacles)

    return inner_idx, new_inner
