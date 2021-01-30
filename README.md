# Boid School Simulation

Simulation of schooling fish with predators.

The following video shows how to play around with the simulator (https://www.youtube.com/watch?v=AvthW_AF5FY):

[![Watch the video](https://img.youtube.com/vi/AvthW_AF5FY/maxresdefault.jpg)](https://www.youtube.com/watch?v=AvthW_AF5FY)

Poster of the research we did using this simulation:

![Poster Image](https://user-images.githubusercontent.com/10909323/106357854-06827300-6309-11eb-975f-6a5a06ed1d0d.png)


## Setup

Our code runs on python3 and uses `pipenv`[1] to manage dependencies, you may need to install both. Not everything may work as expected on platforms other than Linux (in particular, ubuntu).

The following command will install all dependencies (through `pip`):

`pipenv install`

After the command succeeds you should be able to use the program without any issues (on Linux).

[1] https://pypi.org/project/pipenv/


## Visualizing a single simulation

The following command will run a single simulation and generate some plots about the progress of the simulation.

`pipenv run python run.py --plot`

You will see a plot of the progress of the simulation at the end.

To run a few simulations with the same parameters:

`pipenv run python run.py logs/test_logs 4 --plot`

This will run 4 simulations and save the logs of the simulations in `logs/test_logs` (and plot the visualizations afterwards).

To see more information on these simulations, you can view the `Visualise.ipynb` notebook:

`pipenv run jupyter nbconvert --execute --to html Visualize.ipynb`

(or open jupyter notebook from the virtual env)

You can speed up the simulations by setting `sync=false` or `headless=true` in `config.ini`


## Reproducing results

This section shows how to quickly reproduce the results of our research. It takes a long time to run all the simulations however, so we recommend running a single simulation instead, see the section [Visualizing a single simulation]. If you do decide to run all the simulations, make sure to set `headless=true` in `config.ini` to disable the visualization and speed up the simulations significantly.

First, run the simulations:

`pipenv run python run_research.py`

This will take a long time (we ran a lot of simulations). You can edit the `n_sims` variable in `run_research.py` to run fewer. If you want to stop the simulations you need to pause the process (`Ctrl+Z` on linux) and then kill it (`kill %1`).

Then, generate the analysis on the simulations:

`pipenv run jupyter nbconvert --execute --to html VisualizeResearch.ipynb`

(or open jupyter notebook from the virtual env)

You can open the generated html file (`VisualizeResearch.html`) with any browser.

## Tests

We have written no unit tests. There are, however, assertions in the code to ensure the reliability of the simulations. In particular, the file `pars.json` is guaranteed to match the parameters of all simulation logs inside the same directory (by default the log directories are inside `logs/`).
