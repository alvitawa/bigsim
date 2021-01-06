from boid import *
from data import *

import time
import timeit

if __name__ == '__main__':
    from game import *

    # Parameters
    boid_count = 1000
    environment_size = [10, 7]
    grid_size = [1.0, 1.0]

    iterations = 10000

    boid_speed = 0.05
    rotation_rate = 0.95

    # Init population
    n_boids = 1
    population = Population(boid_count, environment_size, boid_speed, grid_size, rotation_rate)

    # Init pygame
    screen = init_pygame(resolution=[1920, 1080])

    tic = time.perf_counter()

    # Simulation loop!
    for i in range(iterations):
        success = draw_population(population, screen)

        # draw_population(population, screen)

        population.iterate(1)

        # pygame.time.delay(100)

        if not success:
            break

    toc = time.perf_counter()

    print(f"Rendered 1000 iterations in {toc - tic:0.4f} seconds")

    # Clean up
    exit_pygame()
