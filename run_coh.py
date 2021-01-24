import numpy as np
import os

from lib.config import cfg

ratios = np.linspace(0, 1, 20, endpoint=True)

python = cfg.get("python")


total_weight = 0.14 + 0.28
# (
#     simulation.pars.alignment_weight + simulation.pars.cohesion_weight
# )

if __name__ == "__main__":
    if not os.path.isdir("./logs/"):
        os.mkdir("./logs")

    while True:
        for ratio in ratios:

            alignment_weight = (1 - ratio) * total_weight
            cohesion_weight = (ratio) * total_weight

            # Pass on the parameters as command-line arguments that will overwrite the default parameters.
            command = f"{python} run.py logs/coh{ratio} 1 --cohesion_weight {cohesion_weight} --alignment_weight {alignment_weight}"
            print(command)
            os.system(command)
