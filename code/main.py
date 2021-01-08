from boid import *
from data import *

import time
import timeit

if __name__ == '__main__':
    from game import *

    # Parameters
    boid_count = 300
    environment_size = [10, 7]
    grid_size = [1.5, 1.5]

    iterations = 100000

    boid_speed = 0.05
    rotation_rate = 0.45

    fps = 60

    # Init population
    n_boids = 1
    population = Population(boid_count, environment_size, boid_speed, grid_size, rotation_rate)

    # Init pygame
    screen, clock = init_pygame(resolution=[1000, 1000])

    tic = time.perf_counter()

    # Simulation loop!
    for i in range(iterations):
        quit = check_input()

        clear_screen(screen)

        draw_population(population, screen)
        draw_sliders()

        update_screen()

        # draw_population(population, screen)

        population.iterate(1)

        clock.tick(fps)

        if quit:
            break

    toc = time.perf_counter()

    print(f"Rendered 1000 iterations in {toc - tic:0.4f} seconds")

    # Clean up
    exit_pygame()
