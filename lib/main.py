from .simulation import *
from .game import *

import pdb
import sys
import time
import timeit
import pickle
import json
import threading
from IPython import embed


from warnings import filterwarnings

from .config import *

stop = False
exc = None
threads = 6

def simulation_loop(simulation, screen, clock, fps):
    start = time.time()

    global stop

    iterations = 0

    computation_time = 0
    render_time = 0

    # Simulation loop!
    big_tic = time.perf_counter()
    with Pool(processes=threads) as pool:
        while not stop:

            quit = check_input()

            tic = time.perf_counter()  # Rendering
            clear_screen(screen)
            
            debug_draw(screen)
            draw_population(screen)
            draw_sliders()

            draw_buttons()

            # Fps counter
            draw_number(screen, int(clock.get_fps()), (0,0), np.abs(np.array(OCEAN_COLOR)-255))

            # Population counter
            draw_number(screen, simulation.population.shape[0], (0.9*WIDTH, 0), np.abs(np.array(OCEAN_COLOR)-200))
            

            # Flip buffers
            update_screen()

            toc = time.perf_counter()

            render_time += toc - tic

            tic = time.perf_counter()  # Computation
            quit = not simulation.iterate(pool, 1) or quit
            toc = time.perf_counter()

            computation_time += toc - tic

            clock.tick(fps)

            if quit:
                break

            iterations += 1
            if iterations >= iterations_left:
                break

    big_toc = time.perf_counter()

    diff = big_toc - big_tic
    
    end = time.time()
    duration = np.array(end)- np.array(start)
    simulation.stats.duration = duration

    simulation.log()
    print("Logged simulation statistics.")

    print(
        f"Rendered {iterations} iterations in {render_time:0.4f} seconds ({render_time/diff*100:0.1f}%). {iterations/render_time:0.4f} iterations/sec"
    )
    print(
        f"Calculated {iterations} iterations in {computation_time:0.4f} seconds ({computation_time/diff*100:0.1f}%). {iterations/computation_time:0.4f} iterations/sec"
    )
    print(
        f"Total {iterations} iterations in {diff:0.4f} seconds. {iterations/diff:0.4f} iterations/sec, (Other expenses were: {diff- render_time - computation_time :0.4f} seconds)"
    )
    print(
        f"{iterations_left}, {grid_size}, {threads}, {computation_time/iterations:0.4f}, {render_time/iterations:0.4f}"
    )
    print("This simulation took ", duration, "seconds")


def exception_catcher(f, *args, **kwargs):
    global exc

    try:
        f(*args, **kwargs)
    except Exception as e:
        exc = sys.exc_info()
        print(e)


def start():
    global simulation_loop, simulation, screen, clock, fps
    thread = threading.Thread(
        target=exception_catcher, args=(simulation_loop, simulation, screen, clock, fps)
    )
    thread.start()

def debug():
    pdb.post_mortem(exc[2])


def pars():
    global simulation
    return simulation.pars

if __name__ == "__main__":

    size = 10

    sight = 3
    global grid_size
    grid_size = 2.5

    box_sight = np.ceil(sight / grid_size)

    # Parameters

    iterations_left = 100000000

    fps = 30

    # Init simulation
    simulation = Simulation(
        None,
        grid_size=(grid_size, grid_size),
        box_sight_radius=box_sight,
        multithreaded=not IPYTHON_MODE,
        default_save=DEFAULT_SAVE
    )

    # Init pygame
    screen, clock = init_pygame(resolution=[WIDTH, HEIGHT], simulation=simulation, do_sliders=SLIDERS)

    if not IPYTHON_MODE:
        simulation_loop(simulation, screen, clock, fps)
    else:
        filterwarnings('ignore')

        start()
        embed()

    stop = True

    # Clean up
    exit_pygame()