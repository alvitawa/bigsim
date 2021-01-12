from sys import is_finalizing
from timeit import default_timer
from typing import Any, Callable
from dataclasses import field

from numpy.lib.function_base import select
from boid import Boid
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Gekke optimalizatie
from multiprocessing import Pool  # , Process, Barrier
from scipy.spatial import distance_matrix
from functools import partial

from dataclasses import dataclass
from dataclasses_json import config, DataClassJsonMixin, dataclass_json

selected_index = None

def find_eaten_fish(distances):
    loca = np.where(distances<0.3)
    indx = loca[0]
    return np.unique(indx)



def delete_fish(population, indexes):
    dead_fish = population[indexes]
    alive_fish = np.delete(population, indexes, 0)
    # delete

    # updated selected fish
    global selected_index
    if selected_index != None:
        if selected_index in indexes:
            selected_index = None
        else:
            selected_index -= sum(indexes < selected_index)

    return alive_fish


@dataclass_json
@dataclass
class Parameters():
    shape: Any = (10, 7)
    boid_count: int = 300
    shark_count: int = 5

    speed: float = 0.05
    agility: float = 0.2

    separation_weight: float = 1.4
    separation_range: float = 0.2

    cohesion_weight: float = 0.05
    cohesion_range: float = 1.0

    alignment_weight: float = 1.0
    alignment_range: float = 0.8

    obstacle_weight: float = 2
    obstacle_range: float = 0.1

    shark_weight: float = 1.8
    shark_range: float = 0.7

    shark_speed: float = 0.05
    shark_agility: float = 0.09

    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)

    # def __post_init__(self):
    #     self.shape = np.array(self.shape)


def generate_population(n, env_size):
    population = np.random.rand(n, 2, 2)
    population[:, 0, :] *= env_size

    population[:, 1, :] -= 0.5
    population[:, 1, :] /= np.linalg.norm(population[:, 1, :], axis=1)[:, None]

    return population


def generate_obstacles(n, env_size):

    x_zeros = np.zeros(n)
    y_zeros = np.zeros(n)

    x_maxes = np.array([env_size[0]] * n)
    y_maxes = np.array([env_size[1]] * n)

    x_coords = np.linspace(0, env_size[0], n)
    y_coords = np.linspace(0, env_size[1], n)

    wtf = np.hstack(
        [
            np.dstack((x_zeros, y_coords)),
            np.dstack((x_maxes, y_coords)),
            np.dstack((x_coords, y_zeros)),
            np.dstack((x_coords, y_maxes)),
        ]
    )

    obstacles = wtf.reshape([len(wtf[0]), 1, 2])

    return obstacles


class Simulation:
    """
        This class is for all boids.
    """

    def __init__(
        self,
        pars=Parameters(),
        grid_size=(1.0, 1.0),
        box_sight_radius=2,
        multithreaded=True,
        default_save='saved_parameters.json'
    ):
        # Save simulation parameters
        self.pars = pars

        # Algo settings
        self.box_sight_radius = box_sight_radius
        self.grid_size = np.array(grid_size)
        self.multithreaded = multithreaded
        self.default_save = default_save

        x_boxes = int(np.ceil(self.pars.shape[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.pars.shape[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(self.pars.boid_count, self.pars.shape)

        # make sharks
        self.sharks = generate_population(self.pars.shark_count, self.pars.shape)

        # make obstacles
        self.obstacles = generate_obstacles(30, self.pars.shape)

        
    def load(self, f=None):
        if f == None:
            f = self.default_save
        with open(f, 'r') as file:
            self.pars = Parameters.from_json(file.read())
            return self.pars

    def save(self, f=None):
        if f == None:
            f = self.default_save
        with open(f, 'w') as file:
            return json.dump(self.pars.to_dict(),  file, indent=4, sort_keys=True)

    def iterate(self, pool, n=1):
        for _ in range(n):
            grid_coordinates = self.population[:, 0, :] // self.grid_size

            results = []
            if self.multithreaded:
                results = pool.map(
                    partial(
                        task,
                        population=self.population,
                        grid_coordinates=grid_coordinates,
                        box_sight_radius=self.box_sight_radius,
                        pars=self.pars,
                        obstacles=self.obstacles,
                        sharks=self.sharks,
                    ),
                    self.boxes,
                )
            else:
                for box in self.boxes:
                    idx, new = task(
                        box,
                        self.population,
                        grid_coordinates,
                        self.box_sight_radius,
                        self.pars,
                        self.obstacles,
                        self.sharks,
                    )
                    results.append((idx, new))

            for idx, new in results:
                self.population[idx] = new

            # wrapping
            self.population[:, 0, 0] %= self.pars.shape[0]
            self.population[:, 0, 1] %= self.pars.shape[1]


            self.sharks, eaten_fish_indexes = move_sharks(self.sharks, self.population, self.obstacles, self.pars)

            self.sharks[:, 0, 0] %= self.pars.shape[0]
            self.sharks[:, 0, 1] %= self.pars.shape[1]

            self.population = delete_fish(self.population, eaten_fish_indexes)

            # solid walls
            # self.population[:, 0, 0] = np.clip(self.population[:, 0, 0], 0, self.pars.shape[0])
            # self.population[:, 0, 1] = np.clip(self.population[:, 0, 1], 0, self.pars.shape[1])

        # with Pool(processes=4) as pool:
        #     results = pool.map(task, parameters)


def stable_norm(array):
    """
    Makes it 0 if not finite
    """
    lengths = np.linalg.norm(array, axis=1)

    norm = np.zeros_like(lengths, dtype=float)
    norm[lengths != 0] = 1.0 / lengths[lengths != 0]

    normed = array * norm[:, None]
    normed[np.invert(np.isfinite(normed))] = 0
    return normed

def fish_move_vectors(fish, neighbours, obstacles, sharks, pars: Parameters):
    # --- Fish Schooling ---
    neighbours_rel = neighbours[:, None, 0, :] - fish[:, 0, :]
    distances = np.sqrt(np.power(neighbours_rel, 2).sum(axis=-1))

    # Cohesion: move to weighted center of mass of school
    cohesion_weights = stats.norm.pdf(distances / (pars.cohesion_range*2)) if (pars.cohesion_range != 0) else np.zeros_like(distances)
    center_off_mass = (neighbours_rel * cohesion_weights[:, :, None]).sum(axis=0)

    # Seperation: move away from very close fish
    seperation_weights = stats.norm.pdf(distances / (pars.separation_range*2)) if (pars.separation_range != 0) else np.zeros_like(distances)
    move_away_target = -1 * (neighbours_rel * seperation_weights[:, :, None]).sum(axis=0)

    # Alignment: align with nearby fish
    alignment_weights = stats.norm.pdf(distances / (pars.alignment_range*2)) if (pars.alignment_range != 0) else np.zeros_like(distances)
    target_alignment = (neighbours[:, None, 1, :] * alignment_weights[:, :, None]).sum(axis=0)

    # --- Obstacles ---
    obstacles_rel = obstacles - fish[:, 0, :]
    obs_distances = np.sqrt(np.power(obstacles_rel, 2).sum(axis=-1))

    obstacle_weights = stats.norm.pdf(obs_distances / (pars.obstacle_range*2)) if (pars.obstacle_range != 0) else np.zeros_like(distances)
    obstacle_target = -1 * (obstacles_rel * obstacle_weights[:, :, None]).sum(axis=0)

    wall_target = None
    # --- Predators ---
    sharks_rel = sharks[:, None, 0, :] - fish[:, 0, :]
    shark_distances = np.sqrt(np.power(sharks_rel, 2).sum(axis=-1))

    shark_weights = stats.norm.pdf(shark_distances / (pars.shark_range*2)) if (pars.shark_range != 0) else np.zeros_like(distances)
    sharks_target = -1 * (sharks_rel * shark_weights[:, :, None]).sum(axis=0)
    # We could also do like turn away from the direction of the shark


    # Normalize directions and weigh them
    cohesion = np.nan_to_num(center_off_mass * pars.cohesion_weight)
    seperation = np.nan_to_num(move_away_target * pars.separation_weight)
    alignment = np.nan_to_num(target_alignment * pars.alignment_weight)

    obstacle = np.nan_to_num(obstacle_target * pars.obstacle_weight)
    shark = np.nan_to_num(sharks_target * pars.shark_weight)

    return cohesion, seperation, alignment, obstacle, shark

def move_fish(fish, neighbours, obstacles, sharks, pars: Parameters):
    # --- Get vectors ---
    cohesion, seperation, alignment, obstacle, shark = fish_move_vectors(fish, neighbours, obstacles, sharks, pars)

    # Combine them to make the steering direction
    vectors = np.array([cohesion, seperation, alignment, obstacle, shark])

    steer_direction = sum(vectors)  # this would be nicer with np.sum(some_axis)
    steer_normed = steer_direction / np.linalg.norm(steer_direction, axis=1)[:, None]

    # print("Steer: ", steer_normed.shape)

    # Combine current direction and steering direction
    updated_fish = np.copy(fish)

    new_direction = fish[:, 1, :] + steer_normed * pars.agility
    # print("New Dir: ", new_direction.shape)
    updated_fish[:, 1, :] = (
        new_direction / np.linalg.norm(new_direction, axis=1)[:, None]
    )

    # move da fish
    updated_fish[:, 0, :] += updated_fish[:, 1, :] * pars.speed

    # check for error
    nans = np.argwhere(np.isnan(updated_fish))
    if nans.shape[0] > 0:
        raise Exception(f"{nans.shape[0]} NaN's encountered in move_fish")

    return updated_fish


def move_sharks(sharks, fish, obstacles, pars: Parameters):
    # Chase: move to weighted center of mass of fish
    fish_rel = fish[:, None, 0, :] - sharks[:, 0, :]
    distances = np.sqrt(np.power(fish_rel, 2).sum(axis=-1))
    
    # TODO: DIFFERENT PARAMETERS for SHARKS
    fish_weights = stats.norm.pdf(distances / (pars.cohesion_range*2))  if (pars.cohesion_range != 0) else np.zeros_like(distances)
    center_off_mass = (fish_rel * fish_weights[:, :, None]).sum(axis=0)

    # Todo: we could also add obstacle avoidance etc.

    # --- Combine vectors ---

    # Normalize directions and weigh them
    chase = np.nan_to_num(center_off_mass * pars.cohesion_weight)

    # Combine them to make the steering direction
    vectors = np.array([chase])

    steer_direction = sum(list(vectors))  # this would be nicer with np.sum(some_axis)
    steer_normed = steer_direction / np.linalg.norm(steer_direction, axis=1)[:, None]

    # print("Steer: ", steer_normed.shape)

    # Combine current direction and steering direction
    updated_shark = np.copy(sharks)

    new_direction = sharks[:, 1, :] + steer_normed * pars.shark_agility
    # print("New Dir: ", new_direction.shape)
    updated_shark[:, 1, :] = (
        new_direction / np.linalg.norm(new_direction, axis=1)[:, None]
    )

    # move da fish
    updated_shark[:, 0, :] += updated_shark[:, 1, :] * pars.shark_speed


    # Eating
    eaten_fish_indexes = find_eaten_fish(distances)

    return updated_shark, eaten_fish_indexes



def task(
    assigned_box,
    population,
    grid_coordinates,
    box_sight_radius,
    pars,
    obstacles,
    sharks,
):
    inner_idx = np.all(np.equal(grid_coordinates, assigned_box.T), axis=1)

    outer_idx = (
        np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= box_sight_radius
    )

    new_inner = move_fish(
        population[inner_idx], population[outer_idx], obstacles, sharks, pars
    )

    return inner_idx, new_inner
