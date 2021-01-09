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
threads = 8

def simulation_loop(population, screen, clock, fps):
    global stop


    iterations = 0

    computation_time = 0
    render_time = 0

    # Simulation loop!
    with Pool(processes=threads) as pool:
        while not stop:
            quit = check_input(population)

            tic = time.perf_counter() # Rendering
            clear_screen(screen)
            draw_population(population, screen)
            draw_sliders()
            update_screen()
            toc = time.perf_counter()

            render_time += toc - tic

            tic = time.perf_counter() # Computation
            population.iterate(pool, 1)
            toc = time.perf_counter()

            computation_time += toc - tic

            clock.tick(fps)

            if quit:
                break

            iterations += 1
    
    print(f"Rendered {iterations} iterations in {render_time:0.4f} seconds. {iterations/render_time:0.4f} iterations/sec")
    print(f"Calculated {iterations} iterations in {computation_time:0.4f} seconds. {iterations/computation_time:0.4f} iterations/sec")

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

    size = 14

    sight = 1
    grid_size = 0.5

    box_sight = np.ceil(sight / grid_size)

    # Parameters
    env = EnvParameters(boid_count=100, shape=(size, size))
    boid = BoidParameters()

    iterations_left = 10000

    fps = 60

    # Init population
    population = Population(env, boid, grid_size=(grid_size, grid_size), box_sight_radius=box_sight)

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
