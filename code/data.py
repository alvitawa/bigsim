import numpy as np
import matplotlib.pyplot as plt

# Gekke optimalizatie
from multiprocessing import Pool, Process, Barrier
from scipy.spatial import distance_matrix


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

    def __init__(self, n, size, movement_speed, grid_size, rotation_rate):

        # boid settings
        self.size = np.array(size)
        self.movement_speed = movement_speed
        self.rotation_rate = rotation_rate

        self.box_sight_radius = 2

        # sim settings
        self.grid_size = np.array(grid_size)

        x_boxes = int(np.ceil(self.size[0] / self.grid_size[0]))
        y_boxes = int(np.ceil(self.size[1] / self.grid_size[1]))

        self.boxes = [np.array([x, y]) for x in range(x_boxes) for y in range(y_boxes)]

        # make population
        self.population = generate_population(n, self.size)

    def iterate(self, n):
        grid_coordinates = self.population[:, 0, :] // self.grid_size

        barrier = Barrier(len(self.boxes))

        results = []
        for box in self.boxes:
            idx, new = task(box, self.population, grid_coordinates, self.box_sight_radius, self.movement_speed, self.rotation_rate) # TODO MULTITHREAD MY ASS
            results.append((idx, new))

        for idx, new in results:
            self.population[idx] = new
        
        # wrapping
        self.population[:, 0, 0] %= self.size[0]
        self.population[:, 0, 1] %= self.size[1]

        # with Pool(processes=4) as pool:
        #     results = pool.map(task, parameters)

# def f(i):
#    return i*i

def filtered_means(outer, pr, distances, min, max):
    mask = (0.5 < distances) & (distances < 1.0)
    # np.fill_diagonal(far, 0)

    filtered_boids = outer[:, None, pr, :] * mask[:, :, None]

    means = filtered_boids.mean(axis=0)

    return means

def norm_v(vector):
    return vector/np.linalg.norm(vector)

def update_zone(inner, outer, speed, rotation_rate):
    distances = distance_matrix(outer[:, 0, :], inner[:, 0, :])

    # Close
    mask = (distances < 1)
    # np.fill_diagonal(far, 0)

    filtered_boids = outer[:, None, 0, :] * mask[:, :, None]
    close_means = filtered_boids.mean(axis=0)

    filtered_boids = outer[:, None, 1, :] * mask[:, :, None]
    close_directions_mean = filtered_boids.mean(axis=0)
    
    # Far
    mask = (1 < distances) & (distances < 4)
    
    filtered_boids = outer[:, None, 0, :] * mask[:, :, None]
    far_means = filtered_boids.mean(axis=0)

    # Rule vectors
    cohesion_vectors = far_means - inner[:, 0, :]
    alignment_vectors = close_directions_mean
    separation_vectors = inner[:, 0, :] - close_means
    
    vectors = [cohesion_vectors, alignment_vectors, separation_vectors]
    weights = [1, 2, 2]

    deltas = sum(w * v / np.linalg.norm(v, axis=1)[:, None] for v, w in zip(vectors, weights))
    deltas /= np.linalg.norm(deltas, axis=1)[:, None]

    updated_inner = np.copy(inner)
    updated_inner[:, 1, :] += rotation_rate * deltas
    updated_inner[:, 1, :] /= np.linalg.norm(updated_inner[:, 1, :], axis=1)[:, None]
    updated_inner[:, 0, :] += speed * updated_inner[:, 1, :]

    return updated_inner
    
def task(assigned_box, population, grid_coordinates, box_sight_radius, speed, rotation_rate):
    inner_idx = np.all(np.equal(grid_coordinates, assigned_box.T), axis=1)

    outer_idx = np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= box_sight_radius

    new_inner = update_zone(population[inner_idx], population[outer_idx], speed, rotation_rate)

    return inner_idx, new_inner
