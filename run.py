"""
    Script to run the simulations.

    Usage: `python run.py [log directory] [number of simulations] ...simulation parameters...`

    Type `python run.py --help` to see all options, including the simulation parameters.

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


def run_test(log_dir, simulation_parameters):
    """
        Create, run and log a single simulation, visualizing it as configured in config.ini

        Returns True if the user exited the application (by closing the visualization)
    """

    # Init simulation
    simulation = Simulation(
        pars=simulation_parameters,
        grid_size=(GRID_SIZE, GRID_SIZE),
        box_sight_radius=BOX_SIGHT,
        n_threads=cfg.getint("n_threads"),
        default_save=cfg.get("save"),
    )

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


def run_multiple_tests(log_dir, n_sims, simulation_parameters):
    """
        Run `n_sims` simulations. Save logs to `log_dir`.

        If there are already logs in log_dir, this function will continue
        appending more logs.
    """
    if log_dir is not None:
        print("Working on: ", log_dir, f" ({n_sims} simulations)")

    for _ in range(n_sims):
        run_test(log_dir, simulation_parameters)

        # Stop if the game window was closed
        if not HEADLESS and game._stop:
            break

    print("Done!")


if __name__ == "__main__":
    import argparse

    from dataclasses import fields

    # There will be one command line option per parameter
    parameters = [field.name for field in fields(Parameters)]

    # Standard command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run N simulations with the configured parameters."
    )
    parser.add_argument(
        "log_directory",
        default=None,
        type=str,
        nargs="?",
        help="The directory to store the logs for this simulations (like logs/test_sims).",
    )
    parser.add_argument(
        "N", default=1, type=int, nargs="?", help="Number of simulations."
    )
    for par in parameters:
        parser.add_argument(f"--{par}", type=float, nargs="?")

    args = parser.parse_args()
    log_dir = args.log_directory
    n_sims = args.N

    try:
        simulation_parameters = Parameters.load(cfg.get("save"))
    except FileNotFoundError:
        simulation_parameters = Parameters()

    for par in parameters:
        value = getattr(args, par)
        if value is None:
            continue
        simulation_parameters[par] = value

    # Run the actual simulations with the specified parameters
    run_multiple_tests(log_dir, n_sims, simulation_parameters)

    if not HEADLESS:
        game.quit()
