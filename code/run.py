from boid import *
from simulation import *
from game import *
from config import *
from ipython_help import *

import time
import timeit
import pickle
import json
import threading

# Parameters
fps = 30

def run_until_dead(simulation):
    # Init pygame
    screen, clock = None, None
    if not HEADLESS:
        screen, clock = init_pygame(resolution=[cfg.getint("width"), cfg.getint("height")], simulation=simulation, do_sliders=cfg.getboolean("sliders"))

    # Iterate until done
    with Pool(processes=cfg.getint("n_threads")) as pool:
        user_exit = False
        for i in range(cfg.getint("max_steps")):
            if not HEADLESS:
                user_exit = visualize(simulation, screen, clock)
                if user_exit:
                    break

            simulate_step(simulation, pool)
            if len(simulation.population) == 0: # TODO: I think it crashes internally when this happens
                break

    # Exit pygame
    if not HEADLESS:
        exit_pygame()

def run_single_simulation():
    # Init simulation
    simulation = Simulation(
        pars=None,
        grid_size=(GRID_SIZE, GRID_SIZE),
        box_sight_radius=BOX_SIGHT,
        multithreaded=not cfg.getboolean("ipython"),
        default_save=cfg.get("save")
    )

    run_until_dead(simulation)

def visualize(simulation, screen, clock):
    quit = check_input()

    clear_screen(screen)
    
    # draw population
    debug_draw(screen)
    draw_population(screen)

    # draw UI
    draw_sliders()
    draw_buttons()

    # draw FPS counter
    draw_number(screen, int(clock.get_fps()), (0,0), np.abs(np.array(OCEAN_COLOR)-255))

    # draw Population counter
    draw_number(screen, simulation.population.shape[0], (0.9*cfg.getint("width"), 0), np.abs(np.array(OCEAN_COLOR)-200))

    update_screen()
    
    clock.tick(fps)

    return quit

def simulate_step(simulation, pool):
    simulation.iterate(pool, 1)

if __name__ == "__main__":
    # Run Simulation
    if not cfg.getboolean("ipython"):
        run_single_simulation()
    else:
        thread = threading.Thread(
            target=exception_catcher, args=(run_single_simulation)
        )
        thread.start()