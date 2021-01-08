from boid import *
from data import *

import time
import timeit

if __name__ == "__main__":
    from game import *

    # Parameters
    env = EnvParameters(boid_count=300, shape=(10, 10))
    boid = BoidParameters()
    # boid.pos_wf = gaussian_pos_wf
    # boid.dir_wf = gaussian_dir_wf

    iterations = 10000

    fps = 60

    # Init population
    population = Population(env, boid)

    # Init pygame
    screen, clock = init_pygame(resolution=[1000, 1000])
    
    tic = time.perf_counter()

    # Simulation loop!
    for i in range(iterations):
        quit = check_input(population)

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
