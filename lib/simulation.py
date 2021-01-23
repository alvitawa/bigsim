import os
import numpy as np
import time

from tqdm import tqdm
from scipy import stats

# Gekke optimalizatie
from multiprocessing import Pool  # , Process, Barrier
from scipy.spatial import distance_matrix
from functools import partial

from .config import CLUSTERING_METHOD
from .cluster import get_clusterer
from .parameters import Parameters
from .statistics import Statistics


def find_eaten_fish(distances):
    loca = np.where(distances < 0.2)
    indx_fish = loca[0]
    indx_shark = loca[1]
    return np.unique(indx_fish), np.unique(indx_shark)


def far_away_sharks(distances):
    loca = np.where(distances > 0.4)
    indx_shark = loca[1]
    return np.unique(indx_shark)


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


def distance_to_weights(sqr_distances, range):
    if range == 0:
        return np.zeros_like(sqr_distances)
    return np.exp(-(sqr_distances / (range / 3.0) ** 2))


class Simulation:
    """
        This class handles the simulation.

        The progress the simulation is in theory solely dependent on the 'pars' argument
        passed to the __init__ function as well as the initial state of the population 
        (which is generated in the __init__ function). However, some other arguments to the
        __init__ function that are meant to fine-tune performance may have an impact as well.
    """

    def __init__(
        self,
        pars=None,
        grid_size=(1.0, 1.0),
        box_sight_radius=2,
        n_threads=4,
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

        self.stats = Statistics()

        # Algo settings
        self.box_sight_radius = box_sight_radius
        self.grid_size = np.array(grid_size)
        self.n_threads = n_threads

        self.selected_index = None
        self.leaders = []

        x_boxes = int(np.ceil(self.pars.shape[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.pars.shape[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(self.pars.boid_count, self.pars.shape)

        # make sharks
        self.sharks = generate_population(self.pars.shark_count, self.pars.shape)
        self.shark_state = np.zeros(self.pars.shark_count)

        # make obstacles
        self.obstacles = generate_obstacles(0, self.pars.shape)

        self.clusterer = get_clusterer(self, self.pars.cluster_method)

        self.labels = -np.ones(self.population.shape[0], dtype=int)

    def load_pars(self, f=None):
        if f == None:
            f = self.default_save
        self.pars = Parameters.load(f)
        return self.pars

    def save_pars(self, f=None):
        if f == None:
            f = self.default_save
        self.pars.save(f)

    def save_stats(self, f):
        self.stats.save(f)

    def log(self, path=None, index=None):
        if path == None:
            if not os.path.isdir("./logs/"):
                os.mkdir("./logs")
            path = "./logs/" + str(time.time())

        pars_path = path + "/pars.json"

        if not os.path.isdir(path):
            os.mkdir(path)
            self.save_pars(pars_path)
        else:
            try:
                pars = Parameters.load(pars_path)
                assert pars == self.pars
            except FileNotFoundError:
                self.save_pars(pars_path)
            except AssertionError:
                raise AssertionError(
                    "Attempted to log simulations with different parameters in the same directory. This is not allowed."
                )

        if index == None:
            # Get highest simulation index in the directory
            high_score = -1
            for file in os.listdir(path):
                parts = file.split(".")

                if parts[0][:5] == "stats":
                    if int(parts[0][5:]) > high_score:
                        high_score = int(parts[0][5:])

            index = high_score + 1

        self.save_stats(f"{path}/stats{index}.json")

    def iterate_once(self, pool):
        """
            Do not use this function directly, use `Simulation.run` (the statistics may become corrupted otherwise)
        """
        grid_coordinates = self.population[:, 0, :] // self.grid_size

        results = []
        if self.n_threads > 1:
            results = pool.map(
                partial(
                    self.get_updated_positions,
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
                idx, new = self.get_updated_positions(
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

        self.sharks, eaten_fish_indexes = self.move_sharks(
            self.sharks, self.population, self.obstacles, self.pars
        )

        self.population = self.delete_fish(self.population, eaten_fish_indexes)

        # solid walls
        self.population[:, 0, 0] = np.clip(
            self.population[:, 0, 0], 0, self.pars.shape[0]
        )
        self.population[:, 0, 1] = np.clip(
            self.population[:, 0, 1], 0, self.pars.shape[1]
        )

        self.sharks[:, 0, 0] = np.clip(self.sharks[:, 0, 0], 0, self.pars.shape[0])
        self.sharks[:, 0, 1] = np.clip(self.sharks[:, 0, 1], 0, self.pars.shape[1])

        if self.stats.iterations % self.pars.resolution == 0:
            self.labels = self.clusterer.fit(self)
            self.stats.measure(self)

        self.stats.iterations += 1

    def run(self, n=None, callback=lambda: None):
        """ 
            Run the simulation for n iterations, or until the end if n is None.

            Returns `True` if the simulation has ended. `False` otherwise.

            The optional argument `callback` will be called before each iteration.
            It must return `True` or `False` and the simulation will stop (but not end)
            as soon as it returns `True`.
        """

        if n == None:
            n = self.pars.max_steps

        with Pool(processes=self.n_threads) as pool:
            start_time = time.time()
            for _ in tqdm(range(n)):
                if (
                    self.population.shape[0] == 0
                    or self.stats.iterations >= self.pars.max_steps <= 0
                ):
                    return True

                if callback():
                    return False

                self.iterate_once(pool)
            end_time = time.time()
            self.stats.duration += end_time - start_time

        return True

    def get_leaders(self):
        return self.leaders

    def update_leaders(self, new_leaders):
        self.leaders = new_leaders

    def fish_move_vectors(self, fish, neighbours, obstacles, sharks, pars: Parameters):
        """
            Calculate a list of directions each fish in the `fish` parameter wants to go
            in. Each direction is represented as a vector, where the magnitude determines
            how badly the fish wants to go in said direction. Per fish there are six 
            different vectors, each for a different strategy like cohesion, separation
            or wall avoidance.
        """
        # --- Fish Schooling ---
        neighbours_rel = neighbours[:, None, 0, :] - fish[:, 0, :]
        sqr_distances = np.sqrt(np.power(neighbours_rel, 2).sum(axis=-1))

        # Cohesion: move to weighted center of mass of school
        cohesion_weights = distance_to_weights(sqr_distances, pars.cohesion_range)
        center_off_mass = (neighbours_rel * cohesion_weights[:, :, None]).sum(axis=0)

        # print(sqr_distances, pars.cohesion_range, cohesion_weights)

        # Seperation: move away from very close fish
        seperation_weights = distance_to_weights(sqr_distances, pars.separation_range)
        move_away_target = -1 * (neighbours_rel * seperation_weights[:, :, None]).sum(
            axis=0
        )

        # Alignment: align with nearby fish
        alignment_weights = distance_to_weights(sqr_distances, pars.alignment_range)
        target_alignment = (
            neighbours[:, None, 1, :] * alignment_weights[:, :, None]
        ).sum(axis=0)

        # --- Obstacles ---
        obstacles_rel = obstacles - fish[:, 0, :]
        sqr_obs_distances = np.sqrt(np.power(obstacles_rel, 2).sum(axis=-1))

        obstacle_weights = distance_to_weights(sqr_obs_distances, pars.obstacle_range)
        obstacle_target = -1 * (obstacles_rel * obstacle_weights[:, :, None]).sum(
            axis=0
        )

        # --- Walls ---
        topleft_target = distance_to_weights(fish[:, 0, :] ** 2, pars.wall_range)
        botright_target = -1 * distance_to_weights(
            (np.array(pars.shape) - fish[:, 0, :]), pars.wall_range
        )

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

        separation = (
            np.nan_to_num(move_away_target) * pars.separation_weight
        )  # * (1 + pars.separation_weight*np.linalg.norm(cohesion, axis=1)[:, None])

        obstacle = np.nan_to_num(obstacle_target * pars.obstacle_weight)
        wall = np.nan_to_num(wall_target * pars.wall_weight)
        shark = np.nan_to_num(sharks_target * pars.shark_weight)

        return cohesion, separation, alignment, obstacle, wall, shark

    def move_fish(self, fish, neighbours, obstacles, sharks, pars: Parameters):
        """
            This is where the logic for fish movement happens. In essence, first calculate the weighed
            average of the move vectors of each fish (weighed by their magnitude), resulting in 
            the direction the fish wants to be going in. Then move the current angle of the 
            fish a fixed percentage in that direction (as determined by the `agility` parameter).
            Lastly, move each fish forward in the direction it is facing.
        """

        # This array will be updated with the new positions for the inner fish
        fish = np.copy(fish, order="C")

        # --- Get vectors ---
        vectors = self.fish_move_vectors(fish, neighbours, obstacles, sharks, pars)

        # The direction each fish wants to go in
        steer_direction = sum(vectors).view(np.complex128)

        # Combine current direction and steering direction using the property
        # of complex numbers that the angle of the product of two complex numbers
        # is the sum of the angles.
        old_direction = fish.view(np.complex128)[:, 1]
        delta = (steer_direction / old_direction) ** (pars.agility)

        new_direction = old_direction * delta
        new_direction /= np.abs(new_direction)

        fish[:, 1, :] = new_direction.view(np.float64)

        # Move the fish forward in the direction they are facing
        fish[:, 0, :] += (
            fish[:, 1, :] * pars.speed
        )

        # Check for NaN's
        nans = np.argwhere(np.isnan(fish))
        if nans.shape[0] > 0:
            raise Exception(f"{nans.shape[0]} NaN's encountered in move_fish")

        return fish

    def delete_fish(self, population, indexes):
        """
            Delete fish from the population, usually when they are eaten.
        """
        # delete
        alive_fish = np.delete(population, indexes, 0)

        # updated selected fish
        if self.selected_index != None:
            if self.selected_index in indexes:
                self.selected_index = None
            else:
                self.selected_index -= sum(indexes < self.selected_index)

        # fix leaders
        if self.leaders != []:
            for i in range(len(self.leaders)):
                if self.leaders[i]:
                    if self.leaders[i] in indexes:
                        self.leaders[i] = None
                    else:
                        self.leaders[i] -= sum(indexes < self.leaders[i])

        # fix labels
        self.labels = np.delete(self.labels, indexes, 0)

        return alive_fish

    def move_sharks(self, sharks, fish, obstacles, pars: Parameters):
        """ 
            The logic behind shark movement is similar to that of fish movement
            (rotate and then move). But their behaviour is not solely described
            by a set of move vectors.
        """
        # Prep
        sharks_rel = sharks[:, None, 0, :] - sharks[:, 0, :]
        shark_distances = np.sqrt(np.power(sharks_rel, 2).sum(axis=-1))

        fish_rel = fish[:, None, 0, :] - sharks[:, 0, :]
        fish_distances = np.sqrt(np.power(fish_rel, 2).sum(axis=-1))

        # Get shark vectors
        # Seperation
        seperation_weights = distance_to_weights(
            shark_distances, pars.shark_separation_range
        )
        move_away_target = -1 * (sharks_rel * seperation_weights[:, :, None]).sum(
            axis=0
        )

        seperation = np.nan_to_num(move_away_target * pars.shark_separation_weight)

        # Center of mass of TOP X FISH
        fish_weights = (
            stats.norm.pdf(fish_distances / (pars.shark_cohesion_range * 2))
            if (pars.shark_cohesion_range != 0)
            else np.zeros_like(fish_distances)
        )
        mass = fish_rel * fish_weights[:, :, None]

        ranking = np.argsort(fish_distances, axis=0)
        # Ignore far away fish to avoid it balancing
        mass[ranking >= self.pars.shark_top_zoveel] = [0, 0]
        # center_off_mass = mass.sum(axis=0)

        # chase_school = np.nan_to_num(center_off_mass * pars.cohesion_weight)

        # Update Shark State
        # ( positive = charging, zero = wonder, negative = cooldown)
        self.shark_state[self.shark_state > 0] -= 1
        self.shark_state[self.shark_state == 1] = -self.pars.shark_cooldown_duration
        self.shark_state[self.shark_state < 0] += 1

        closest_fish_distances = np.min(fish_distances, axis=0)

        sharks_to_start_charging = np.logical_and(
            (self.shark_state == 0),
            (closest_fish_distances < self.pars.shark_chase_range),
        )
        self.shark_state[sharks_to_start_charging] = self.pars.shark_chase_duration

        # Closest Fish
        closest_id = np.argmin(fish_distances, axis=0)
        positions = fish[:, 0]
        clos_pos = positions[closest_id]

        # Vector to closest fish
        chase_close = clos_pos - sharks[:, 0, :]

        # normally chase schools and seperate, but charge closest fish when charging
        # vectors = np.array(sum([chase_school, seperation])) This is how we first did it but that comes out wack
        vectors = np.array(sum([chase_close, seperation]))

        vectors[self.shark_state > 0] = chase_close[self.shark_state > 0]

        # Combine current direction and steering direction

        steer_direction = vectors.view(np.complex128)

        old_direction = sharks.view(np.complex128)[:, 1]
        delta = (steer_direction / old_direction) ** (pars.shark_agility)

        new_direction = old_direction * delta
        new_direction /= np.abs(new_direction)

        sharks[:, 1, :] = new_direction.view(np.float64)

        self.pars.shark_wonder_speed = 0.025
        self.pars.shark_charge_speed = 0.075
        self.pars.shark_eaten_speed = 0.015

        update = np.zeros_like(sharks[:, 1, :])

        update[(self.shark_state == 0)] = (
            sharks[(self.shark_state == 0)][:, 1, :] * self.pars.shark_wonder_speed
        )
        update[(self.shark_state > 0)] = (
            sharks[(self.shark_state > 0)][:, 1, :] * self.pars.shark_charge_speed
        )
        update[(self.shark_state < 0)] = (
            sharks[(self.shark_state < 0)][:, 1, :] * self.pars.shark_eaten_speed
        )

        sharks[:, 0, :] += update

        # Eating
        eaten_fish_indexes, eating_sharks = find_eaten_fish(fish_distances)
        self.shark_state[eating_sharks] = -self.pars.shark_cooldown_duration

        return sharks, eaten_fish_indexes

    def get_updated_positions(
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
