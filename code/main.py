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
threads = 1

def simulation_loop(population, screen, clock, fps):
    global stop


    iterations = 0

    computation_time = 0
    render_time = 0

    fps_measurer = 0
    fpers = 0
    second = 0

    # Simulation loop!
    big_tic = time.perf_counter()
    with Pool(processes=threads) as pool:
        while not stop:
            begin = time.time()

            quit = check_input(population)

            tic = time.perf_counter() # Rendering
            clear_screen(screen)
            draw_population(population, screen)
            draw_sliders()

            # Fps counter
            draw_number(screen, fps_measurer)

            # Flip buffers
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
            if iterations >= iterations_left:
                break
            
            fpers += 1

            # Display fps
            second += time.time() - begin
            if second > 1:
                fps_measurer = fpers
                fpers = 0
                second = 0

    big_toc = time.perf_counter()

    diff = big_toc - big_tic
    
    print(f"Rendered {iterations} iterations in {render_time:0.4f} seconds ({render_time/diff*100:0.1f}%). {iterations/render_time:0.4f} iterations/sec")
    print(f"Calculated {iterations} iterations in {computation_time:0.4f} seconds ({computation_time/diff*100:0.1f}%). {iterations/computation_time:0.4f} iterations/sec")
    print(f"Total {iterations} iterations in {diff:0.4f} seconds. {iterations/diff:0.4f} iterations/sec, (Other expenses were: {diff- render_time - computation_time :0.4f} seconds)")
    print(f"{iterations_left}, {grid_size}, {threads}, {computation_time/iterations:0.4f}, {render_time/iterations:0.4f}")

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

    size = 10

    sight = 3
    global grid_size
    grid_size = 2.5

    box_sight = np.ceil(sight / grid_size)

    # Parameters
    env = EnvParameters(boid_count=200, shape=(size, size))
    boid = BoidParameters()

    iterations_left = 100000000

    fps = 60

    # Init population
    population = Simulation(env, boid, grid_size=(grid_size, grid_size), box_sight_radius=box_sight)

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
