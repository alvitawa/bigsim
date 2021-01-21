from sys import argv
from numpy.lib.scimath import arccos
from lib.simulation import *
from lib.game import *
from lib.config import *
from lib.ipython_help import *

import time
import timeit
import pickle
import json
import threading

# Parameters
fps = 20
user_exit = False

def run_until_max_steps(simulation):
    # Init pygame
    screen, clock = None, None
    if not HEADLESS:
        screen, clock = init_pygame(resolution=[cfg.getint("width"), cfg.getint("height")], simulation=simulation, do_sliders=cfg.getboolean("sliders"))

    # Iterate until done
    with Pool(processes=cfg.getint("n_threads")) as pool:
        global user_exit
        user_exit = False
        while True:
            if not HEADLESS:
                user_exit = visualize(simulation, screen, clock)
                if user_exit:
                    break

            # The simulation itself tracks the remaining iterations
            # and returns zero if it is done
            remaining = simulation.iterate(pool, 1)
            if remaining == 0:
                break

    return len(simulation.population)

def run_test(log_dir, t, cohesion_percent):
    # Init simulation
    simulation = Simulation(
        pars=None,
        grid_size=(GRID_SIZE, GRID_SIZE),
        box_sight_radius=BOX_SIGHT,
        multithreaded=not cfg.getboolean("ipython"),
        default_save=cfg.get("save")
    )

    total_weight = simulation.pars.alignment_weight + simulation.pars.cohesion_weight

    simulation.pars.alignment_weight = (1 - cohesion_percent) * total_weight
    simulation.pars.cohesion_weight = (cohesion_percent) * total_weight

    result = run_until_max_steps(simulation)

    simulation.log(log_dir, t)

    return result

def run_multiple_tests(log_dir, n_sims, ratio):
    for t in range(n_sims):
        high_score = -1
        if (log_dir):
            if os.path.isdir(log_dir):
                for file in os.listdir(log_dir):
                    parts = file.split(".")

                    if parts[0][:5] == "stats":
                        if int(parts[0][5:]) > high_score:
                            high_score = int(parts[0][5:])

        print("Working on: ", log_dir, "/stats", high_score + 1, ".json", sep="")

        run_test(log_dir, high_score + 1, ratio)
        if user_exit:
            break

    print("Done!")

def run_single_simulation(log_dir=None, index=None):
    # Init simulation
    simulation = Simulation(
        pars=None,
        grid_size=(GRID_SIZE, GRID_SIZE),
        box_sight_radius=BOX_SIGHT,
        multithreaded=not cfg.getboolean("ipython"),
        default_save=cfg.get("save")
    )

    run_until_max_steps(simulation)

    simulation.log(log_dir, index)

def visualize(simulation, screen, clock):
    quit = check_input()

    clear_screen(screen)
    
    # draw population
    debug_draw(screen, cfg.getint("max_steps"))
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

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else None
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # Run Simulation
    if not cfg.getboolean("ipython"):
        run_multiple_tests(log_dir, n_sims, ratio)
        exit_pygame()
    else:
        thread = threading.Thread(
            target=exception_catcher, args=(run_single_simulation,)
        )
        thread.start()
        embed()