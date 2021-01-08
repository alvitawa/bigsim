from boid import *
from data import *

import time
import timeit

if __name__ == "__main__":
    from game import *

    # Parameters
    env = EnvParameters(boid_count=100, shape=(10, 7))
    boid = BoidParameters()
    # boid.pos_wf = gaussian_pos_wf
    # boid.dir_wf = gaussian_dir_wf

    iterations = 10000

    # Init population
    population = Population(env, boid)

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
