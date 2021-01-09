from boid import *
from data import *

import sys
import time
import timeit
import threading
from IPython import embed

FUCK_IPYTHON = True

stop = False
exc_info = None
threads = 4

def simulation_loop(population, screen, clock, fps):
    global stop

    tic = time.perf_counter()

    iterations = 0

    # Simulation loop!
    with Pool(processes=threads) as pool:
        while not stop:
            quit = check_input(population)

            clear_screen(screen)

            draw_population(population, screen)
            draw_sliders()

            update_screen()

            draw_population(population, screen)

            population.iterate(pool, 1)

            clock.tick(fps)

            if quit:
                break

            iterations += 1

    toc = time.perf_counter()

    diff = toc - tic
    
    print(f"Rendered {iterations} iterations in {diff:0.4f} seconds. {iterations/diff:0.4f} iterations/sec")

def exception_catcher(f, *args, **kwargs):
    global exc_info

    try:
        f(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)

def start():
    global simulation_loop, population, screen, clock, fps
    thread = threading.Thread(target=exception_catcher, args=(simulation_loop, population, screen, clock, fps))
    thread.start()
        

if __name__ == "__main__":
    from game import *

    # Parameters
    env = EnvParameters(boid_count=100, shape=(14, 8))
    boid = BoidParameters()

    iterations_left = 10000

    fps = 60

    # Init population
    population = Population(env, boid)

    # Init pygame
    screen, clock = init_pygame(resolution=[1400, 800], boid_parameters=boid)

    if FUCK_IPYTHON:
        simulation_loop(population, screen, clock, fps)

    else:
        start()
        embed()

    stop = True
    
    # Clean up
    exit_pygame()
