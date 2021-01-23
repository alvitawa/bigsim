import numpy as np
import os

from lib.config import cfg

ratios = np.linspace(0, 1, 20, endpoint=True)

python = cfg.get("python")

if __name__ == "__main__":
    while True:
        for ratio in ratios:
            print("Working on", ratio)
            os.system(f"{python} run.py logs/alignment{ratio} 2 {ratio}")