from boid import *
from data import *

import pdb
import sys
import time
import timeit
import pickle
import json
import threading
from IPython import embed

import configparser

from warnings import filterwarnings

config = configparser.ConfigParser()
config.read("config.ini")

IPYTHON_MODE = config["DEFAULT"]["ipython"]=='1'
SLIDERS = config["DEFAULT"]["sliders"]=='1'
DEFAULT_SAVE = config["DEFAULT"]["save"]

MENU = False

stop = False
exc = None
threads = 6

def simulation_loop(simulation, screen, clock, fps):
    global stop

    iterations = 0

    computation_time = 0
    render_time = 0

    # Simulation loop!
    big_tic = time.perf_counter()
    with Pool(processes=threads) as pool:
        while not stop:

            quit = check_input(simulation)

            tic = time.perf_counter()  # Rendering
            clear_screen(screen)
            
            debug_draw(simulation, screen)
            draw_population(simulation, screen)
            draw_sliders()

            draw_buttons()

            # Fps counter
            draw_number(screen, int(clock.get_fps()))

            # Flip buffers
            update_screen()

            toc = time.perf_counter()

            render_time += toc - tic

            tic = time.perf_counter()  # Computation
            simulation.iterate(pool, 1)
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
    from game import *

    size = 10

    sight = 3
    global grid_size
    grid_size = 2.5

    box_sight = np.ceil(sight / grid_size)

    # Parameters

    iterations_left = 100000000

    fps = 60

    # Init simulation
    simulation = Simulation(
        None,
        grid_size=(grid_size, grid_size),
        box_sight_radius=box_sight,
        multithreaded=not IPYTHON_MODE,
        default_save=DEFAULT_SAVE
    )

    # Init pygame
    screen, clock = init_pygame(resolution=[980, 600], simulation_pars=simulation.pars, do_sliders=SLIDERS)

    if not IPYTHON_MODE:
        simulation_loop(simulation, screen, clock, fps)
    else:
        filterwarnings('ignore')

        def lp():
            global simulation
            return simulation.load()

        def wp():
            global simulation
            return simulation.save()

        start()
        embed()

    stop = True

    # Clean up
    exit_pygame()
