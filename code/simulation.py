from sys import is_finalizing
from timeit import default_timer
from typing import Any, Callable
from dataclasses import field

from numpy.lib.function_base import select
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats

# Gekke optimalizatie
from multiprocessing import Pool  # , Process, Barrier
from scipy.spatial import distance_matrix
from functools import partial

from dataclasses import dataclass
from dataclasses_json import config, DataClassJsonMixin, dataclass_json

from config import CLUSTERING_METHOD
import cluster


def find_eaten_fish(distances):
    loca = np.where(distances<0.2)
    indx_fish = loca[0]
    indx_shark = loca[1]
    return np.unique(indx_fish), np.unique(indx_shark)

def far_away_sharks(distances):
    loca = np.where(distances>0.4)
    indx_shark = loca[1]
    return np.unique(indx_shark)

@dataclass_json
@dataclass
class Parameters:
    shape: Any = (10, 7)
    boid_count: int = 300
    shark_count: int = 0

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

    separation_range_shark: float = 1
    separation_weight_shark: float = 15

    shark_speed: float = 0.05
    shark_agility: float = 0.09

    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)

    # def __post_init__(self):
    #     self.shape = np.array(self.shape)

@dataclass_json
@dataclass
class Statistics():
    """Number of iterations executed"""
    iterations: int = 0
    """Number of frames (simulation iterations) between measurements"""
    resolution: int = 100
    boid_count: list = field(default_factory=lambda: [])
    school_count: list = field(default_factory=lambda: [])
    school_sizes: list = field(default_factory=lambda: [])
    cluster_method: str = CLUSTERING_METHOD

    def measure(self, sim):
        self.boid_count.append(int(sim.population.shape[0]))
        clusters = np.unique(sim.labels)
        self.school_count.append(int(clusters.shape[0]))
        school_sizes = np.equal(sim.labels[:, None], clusters[None, :]).sum(axis=0)
        school_sizes.sort()
        self.school_sizes.append(list(int(s) for s in school_sizes[::-1]))
        self.iterations += 1

    def schools(self):
        sc = np.array(self.school_count)
        schools = np.zeros(sc.shape + (sc.max(),))
        for i, ss in enumerate(self.school_sizes):
            schools[i, :len(ss)] = ss
        return schools

    def save(self, f):
        with open(f, "w") as file:
            return json.dump(self.to_dict(), file)

    
    def __getitem__(self, index):
        return getattr(self, index)

    def __setitem__(self, index, value):
        setattr(self, index, value)

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

def stable_norm(array):
    """
    Makes it 0 if norm is 0
    """
    norms = 1 / np.linalg.norm(array, axis=1)[:, None]
    norms = np.nan_to_num(norms)

    normed = array * norms
    return normed

class Simulation:
    """
        This class is for all boids.
    """

    def __init__(
        self,
        pars=None,
        stats=Statistics(),
        grid_size=(1.0, 1.0),
        box_sight_radius=2,
        multithreaded=True,
        default_save="saved_parameters.json",
    ):
        self.recently_ate = []
        self.default_save = default_save

        self.pars = pars
        if self.pars == None:
            try:
                self.load_pars()
            except Exception as e:
                print("Couldn't load parameters.")
                print(e)
                self.pars = Parameters()

        self.stats = stats
        # Algo settings
        self.box_sight_radius = box_sight_radius
        self.grid_size = np.array(grid_size)
        self.multithreaded = multithreaded

        self.selected_index = None
        self.leaders = []

        x_boxes = int(np.ceil(self.pars.shape[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.pars.shape[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(self.pars.boid_count, self.pars.shape)

        # make sharks
        self.sharks = generate_population(self.pars.shark_count, self.pars.shape)

        # make obstacles
        self.obstacles = generate_obstacles(0, self.pars.shape)

        
        self.clusterer = cluster.get_clusterer(self, self.stats.cluster_method)

        self.labels = -np.ones(self.population.shape[0], dtype=int)

    def load_pars(self, f=None):
        if f == None:
            f = self.default_save
        with open(f, "r") as file:
            self.pars = Parameters.from_json(file.read())
            return self.pars

    def save_pars(self, f=None):
        if f == None:
            f = self.default_save
        with open(f, "w") as file:
            return json.dump(self.pars.to_dict(), file, indent=4, sort_keys=True)

    def save_stats(self, f):
        self.stats.save(f)

    def log(self, path=None, index=None):
        if path == None:
            path = "logs/" + str(time.time())
            os.mkdir(path)

        self.save_pars(path + "/pars.json")
        indexstr = index if index is not None else ""
        self.save_stats(f"{path}/stats{indexstr}.json")


    def iterate(self, pool, n=1):
        for _ in range(n):
            if self.population.shape[0] == 0:
                return False

            grid_coordinates = self.population[:, 0, :] // self.grid_size

            results = []
            if self.multithreaded:
                results = pool.map(
                    partial(
                        self.task,
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
                    idx, new = self.task(
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
            # self.population[:, 0, 0] %= self.pars.shape[0]
            # self.population[:, 0, 1] %= self.pars.shape[1]


            self.sharks, eaten_fish_indexes = self.move_sharks(self.sharks, self.population, self.obstacles, self.pars)

            # self.sharks[:, 0, 0] %= self.pars.shape[0]
            # self.sharks[:, 0, 1] %= self.pars.shape[1]

            self.population = self.delete_fish(self.population, eaten_fish_indexes)

            # solid walls
            self.population[:, 0, 0] = np.clip(self.population[:, 0, 0], 0, self.pars.shape[0])
            self.population[:, 0, 1] = np.clip(self.population[:, 0, 1], 0, self.pars.shape[1])

            self.sharks[:, 0, 0] = np.clip(self.sharks[:, 0, 0], 0, self.pars.shape[0])
            self.sharks[:, 0, 1] = np.clip(self.sharks[:, 0, 1], 0, self.pars.shape[1])
        
        if self.stats.iterations % self.stats.resolution != 0:
            self.labels = self.clusterer.fit(self)
            self.stats.measure(self)

        self.stats.iterations += 1

        return True

    def get_leaders(self):
        return self.leaders

    def update_leaders(self, new_leaders):
        self.leaders = new_leaders
        return

    def delete_fish(self, population, indexes):
        # dead_fish = population[indexes]
        alive_fish = np.delete(population, indexes, 0)
        # delete
        # updated selected fish
        
        if self.selected_index != None:
            if self.selected_index in indexes:
                self.selected_index = None
            else:
                self.selected_index -= sum(indexes < self.selected_index)

        if self.leaders != []:
            for i in range(len(self.leaders)):
                if self.leaders[i]:
                    if self.leaders[i] in indexes:
                        self.leaders[i] = None
                    else:
                        self.leaders[i] -= sum(indexes < self.leaders[i])
                

        return alive_fish

    def move_sharks(self, sharks, fish, obstacles, pars: Parameters):

        sharks = np.copy(sharks, order='C')
    
        neighbours_rel = sharks[:, None, 0, :] - sharks[:, 0, :]
        sqr_distances = np.sqrt(np.power(neighbours_rel, 2).sum(axis=-1))

        seperation_weights = distance_to_weights(sqr_distances, pars.separation_range_shark)
        move_away_target = -1 * (neighbours_rel * seperation_weights[:, :, None]).sum(axis=0)

        seperation = np.nan_to_num(move_away_target * pars.separation_weight_shark)

        # Chase: move to weighted center of mass of fish
        fish_rel = fish[:, None, 0, :] - sharks[:, 0, :]
        distances = np.sqrt(np.power(fish_rel, 2).sum(axis=-1))
        
        # TODO: DIFFERENT PARAMETERS for SHARKS
        fish_weights = stats.norm.pdf(distances / (pars.cohesion_range*2))  if (pars.cohesion_range != 0) else np.zeros_like(distances) # fuck it use cohesion weight for now
        center_off_mass = (fish_rel * fish_weights[:, :, None]).sum(axis=0)

        # Todo: we could also add obstacle avoidance etc.

        # Closest fish
        closest_id = np.argmin(distances, axis=0)
        positions = fish[:, 0]
        clos_pos = positions[closest_id]
        
        # Vector to closest fish
        chase_close = clos_pos - sharks[:, 0, :]

        # --- Combine vectors ---

        # Normalize directions and weigh them
        chase_school = np.nan_to_num(center_off_mass * pars.cohesion_weight)

        random_threshold = 0.1
        we_go_to_close_fish_or_no = (np.min(distances, axis=0) < random_threshold).astype(int)
        # print(chase_close)
        # print(chase_school)
        # print(we_go_to_close_fish_or_no)
        chase = chase_close * we_go_to_close_fish_or_no.reshape(len(sharks), 1) + np.abs(we_go_to_close_fish_or_no-1).reshape(len(sharks), 1) * chase_school

        # Combine them to make the steering direction
        vectors = np.array([chase, seperation])

        steer_direction = sum(list(vectors)).view(np.complex128)  # this would be nicer with np.sum(some_axis)

        # print("Steer: ", steer_normed.shape)

        # Combine current direction and steering direction
        
        old_direction = sharks.view(np.complex128)[:, 1]
        delta = (steer_direction / old_direction)**(pars.shark_agility)

        new_direction = old_direction * delta
        new_direction /= np.abs(new_direction)

        sharks[:, 1, :] = new_direction.view(np.float64)

        sharks[:, 0, :] += sharks[:, 1, :] * pars.shark_speed


        # Eating
        eaten_fish_indexes = find_eaten_fish(distances)[0]
        eating_sharks = find_eaten_fish(distances)[1]
        self.recently_ate = self.recently_ate + list(eating_sharks)
        

        return sharks, eaten_fish_indexes
        # with Pool(processes=4) as pool:
        #     results = pool.map(task, parameters)

        
    def fish_move_vectors(self, fish, neighbours, obstacles, sharks, pars: Parameters):
        # --- Fish Schooling ---
        neighbours_rel = neighbours[:, None, 0, :] - fish[:, 0, :]
        sqr_distances = np.sqrt(np.power(neighbours_rel, 2).sum(axis=-1))

        # Cohesion: move to weighted center of mass of school
        cohesion_weights = distance_to_weights(sqr_distances, pars.cohesion_range)
        center_off_mass = (neighbours_rel * cohesion_weights[:, :, None]).sum(axis=0)

        # print(sqr_distances, pars.cohesion_range, cohesion_weights)

        # Seperation: move away from very close fish
        seperation_weights = distance_to_weights(sqr_distances, pars.separation_range)
        move_away_target = -1 * (neighbours_rel * seperation_weights[:, :, None]).sum(axis=0)

        # Alignment: align with nearby fish
        alignment_weights = distance_to_weights(sqr_distances, pars.alignment_range)
        target_alignment = (neighbours[:, None, 1, :] * alignment_weights[:, :, None]).sum(axis=0)

        # --- Obstacles ---
        obstacles_rel = obstacles - fish[:, 0, :]
        sqr_obs_distances = np.sqrt(np.power(obstacles_rel, 2).sum(axis=-1))

        obstacle_weights = distance_to_weights(sqr_obs_distances, pars.obstacle_range)
        obstacle_target = -1 * (obstacles_rel * obstacle_weights[:, :, None]).sum(axis=0)

        # --- Walls ---
        topleft_target = distance_to_weights(fish[:, 0, :]**2, pars.wall_range)
        botright_target = -1 * distance_to_weights((np.array(pars.shape) - fish[:, 0, :]), pars.wall_range) 


        wall_target = topleft_target + botright_target

        # --- Predators ---
        sharks_rel = sharks[:, None, 0, :] - fish[:, 0, :]
        sqr_shark_distances = np.sqrt(np.power(sharks_rel, 2).sum(axis=-1))

        shark_weights = distance_to_weights(sqr_shark_distances, pars.shark_range)
        sharks_target = -1 * (sharks_rel * shark_weights[:, :, None]).sum(axis=0)
        # We could also do like turn away from the direction of the shark

        # Weigh directions
        cohesion = stable_norm(center_off_mass) * pars.cohesion_weight
        alignment = stable_norm(target_alignment) * pars.alignment_weight

        separation = np.nan_to_num(move_away_target) * pars.separation_weight #* (1 + pars.separation_weight*np.linalg.norm(cohesion, axis=1)[:, None])

        obstacle = np.nan_to_num(obstacle_target * pars.obstacle_weight)
        wall = np.nan_to_num(wall_target * pars.wall_weight)
        shark = np.nan_to_num(sharks_target * pars.shark_weight)

        return cohesion, separation, alignment, obstacle, wall, shark

    def move_fish(self, fish, neighbours, obstacles, sharks, pars: Parameters):
        """
            Updates the first parameter 'fish'
        """
        
        # This array will be updated with the new positions for the inner fish
        fish = np.copy(fish, order='C')

        # --- Get vectors ---
        vectors = self.fish_move_vectors(fish, neighbours, obstacles, sharks, pars)

        steer_direction = sum(vectors).view(np.complex128)  # this would be nicer with np.sum(some_axis)
        # confidence = np.linalg.norm(steer_direction, axis=1)[:, None]
        # confidence[confidence == 0] = 1
        # steer_normed = steer_direction #/ confidence

        # print("Steer: ", steer_normed.shape)

        # Combine current direction and steering direction

        old_direction = fish.view(np.complex128)[:, 1]
        delta = (steer_direction / old_direction)**(pars.agility)

        new_direction = old_direction * delta
        new_direction /= np.abs(new_direction)

        fish[:, 1, :] = new_direction.view(np.float64)

        # new_direction = fish[:, 1, :] + steer_normed * pars.agility
        # lengths = np.linalg.norm(new_direction, axis=1)[:, None]
        # updated_fish[:, 1, :] = new_direction / lengths

        # move da fish
        fish[:, 0, :] += fish[:, 1, :] * pars.speed #* (1 / (1 + np.exp(-(confidence - 500)/100)) + 1)

        # check for error
        nans = np.argwhere(np.isnan(fish))
        if nans.shape[0] > 0:
            raise Exception(f"{nans.shape[0]} NaN's encountered in move_fish")
        
        return fish



    def task(
        self,
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

        inner_fish = self.move_fish(
            population[inner_idx], population[outer_idx], obstacles, sharks, pars
        )

        return inner_idx, inner_fish


def distance_to_weights(sqr_distances, range):
    if range == 0:
        return np.zeros_like(sqr_distances)
    return np.exp(-(sqr_distances / (range/3.0)**2))
