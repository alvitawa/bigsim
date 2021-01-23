# Boid School Simulation

## Setup

Our code runs on python3 and uses `pipenv`[1] to manage dependencies, you may need to install both.

The following command will install all dependencies (through `pip`):

`pipenv install`

After the command succeeds you should be able to use the program without any issues.

[1] https://pypi.org/project/pipenv/

## Reproducing results

For details on how to use the software in general, read the section [General Usage]. This section shows how to quickly reproduce the results of our research.

First, run the simulations:

`pipenv run python run_coh.py`   

(This may take a while)

Then, generate the analysis on the simulations:

`pipenv run jupyter nbconvert --execute --to html Visualize.ipynb`

You can open the generated html file (`Visualize.html`) with any browser.

## General usage

There are two components to our software architecture:
    1. Generating simulation data
    2. Analyzing simulation data

### Generating simulation data

A single simulation can be run with:

`pipenv run python run.py`

This will use the simulation parameters in `saved_parameters.json` by default.

To run multiple simulations with the same parameters, use:

`pipenv run python logs/multiple_sims 10`

This will run 10 simulations and save the simulation logs in `logs/multiple_sims`

### Analyzing simulation data

There are a few jupyter notebooks in the root directory that show how to make visualizations based on the simulation logs.

Make sure to run jupyter notebook from the pipenv environment:

`pipenv run jupyter notebook`

## Tests

We have written no unit tests. There are, however, assertions in the code to ensure the reliability of the simulations. In particular, the file `pars.json` is guaranteed to match the parameters of all simulation logs inside the same directory (by default the log directories are inside `logs/`).