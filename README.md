# Boid School Simulation

## Setup

Our code runs on python3 and uses `pipenv`[1] to manage dependencies, you may need to install both.

The following command will install all dependencies (through `pip`):

`pipenv install`

After the command succeeds you should be able to use the program without any issues.

[1] https://pypi.org/project/pipenv/

## Reproducing results

This section shows how to quickly reproduce the results of our research. It takes a long time to run all the simulations however, so we recommend running a single simulation instead, see the section [Visualizing a single simulation]. If you do decide to run all the simulations, make sure to set `headless=true` in `config.ini` to disable the visualization and speed up the simulations significantly.

First, run the simulations:

`pipenv run python run_coh.py`   

This will take a long time (we ran a lot of simulations). You can edit the `n_sims` parameters to run fewer.

Then, generate the analysis on the simulations:

`pipenv run jupyter nbconvert --execute --to html CA_Visualize.ipynb`

You can open the generated html file (`Visualize.html`) with any browser.

## Visualizing a single simulation

The following command will run a single simulation and generate some plots about the progress of the simulation.

`pipenv run python run.py --plot`

The plots are stored in Visualize.html, the program will try to open them in a web browser.

To run a few simulations with the same parameters:

`pipenv run python run.py logs/test_logs 4 --plot`

This will run 4 simulations and save the logs of the simulations in `logs/test_logs` (and plot the visualizations afterwards).

You can speed up the simulations by setting `sync=false` or `headless=true` in `config.ini`

## Tests

We have written no unit tests. There are, however, assertions in the code to ensure the reliability of the simulations. In particular, the file `pars.json` is guaranteed to match the parameters of all simulation logs inside the same directory (by default the log directories are inside `logs/`).