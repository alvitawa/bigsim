"""
    Script to run the simulations.

    Usage: python3 [log directory] [number of simulations] [cohesion percentage]

    Global configuration that does not affect the results of the simulatinos
    can be found in the file `config.ini`

    Parameters that do influence the simulation (number of boids, speed of sharks ...etc)
    are found by default in the file `saved_parameters.json`
    
    The name of this file can be changed by editing `config.ini`

    If [cohesion percentage] is supplied, the corresponding values from `save_parameters.json`
    will be overwritten.

"""
from lib.simulation import *
from lib.config import *

import lib.game as game

import threading


def run_test(log_dir=None, cohesion_percent=None):
    """
        Create, run and log a single simulation, visualizing it as configured in config.ini

        Returns True if the user exited the application (by closing the visualization)
    """

    # Init simulation
    simulation = Simulation(
        pars=None,
        grid_size=(GRID_SIZE, GRID_SIZE),
        box_sight_radius=BOX_SIGHT,
        n_threads=cfg.getint("n_threads"),
        default_save=cfg.get("save"),
    )

    if cohesion_percent != None:
        total_weight = (
            simulation.pars.alignment_weight + simulation.pars.cohesion_weight
        )

        simulation.pars.alignment_weight = (1 - cohesion_percent) * total_weight
        simulation.pars.cohesion_weight = (cohesion_percent) * total_weight

    # Initialize the visualization
    if not HEADLESS:
        # Note that the game module is stateful and behaves similarly to a class instance
        game.init(
            resolution=[cfg.getint("width"), cfg.getint("height")],
            simulation=simulation,
            enable_menu=cfg.getboolean("menu"),
            enable_metrics=cfg.getboolean("metrics"),
        )

    # Run the simulation, the callback keeps the visualization synchronized
    simulation.run(callback=game.tick if not HEADLESS else lambda: False)

    if not HEADLESS:
        # One last tick to render the final state of the simulation
        game.tick()

    simulation.log(log_dir)

    if not HEADLESS:
        return game._stop
    return False


def run_multiple_tests(log_dir=None, n_sims=1, ratio=None):
    """
        Run `n_sims` simulations. Save logs to `log_dir`.

        If there are already logs in log_dir, this function will continue
        appending more logs.
    """
    if log_dir is not None:
        print("Working on: ", log_dir, f" ({n_sims} simulations)")

    for _ in range(n_sims):
        user_exit = run_test(log_dir, ratio)

        if user_exit:
            break

    print("Done!")


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else None
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else None

    run_multiple_tests(log_dir, n_sims, ratio)

    if not HEADLESS:
        game.quit()
