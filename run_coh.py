import numpy as np
import os

from lib.config import cfg

ratios = np.linspace(0, 1, 20, endpoint=True)

python = cfg.get("python")


total_weight = 200.14
# (
#     simulation.pars.alignment_weight + simulation.pars.cohesion_weight
# )

if __name__ == "__main__":
    while True:
        for ratio in ratios:

            alignment_weight = (1 - ratio) * total_weight
            cohesion_weight = (ratio) * total_weight

            # Pass on the parameters as command-line arguments that will overwrite the default parameters.
            os.system(f"{python} run.py logs/coh{ratio} 20 --cohesion_weight {cohesion_weight} --alignment_weight {alignment_weight}")
