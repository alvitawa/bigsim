import numpy as np
import os

from lib.config import cfg

ratios = np.linspace(0, 1, 10, endpoint=True)

python = cfg.get("python")

if __name__ == "__main__":
    for ratio in ratios:
        os.system(f"{python} run.py logs/coh{ratio} 20 {ratio}")