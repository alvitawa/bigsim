import numpy as np
import os

ratios = list(np.linspace(0.1,1,8,endpoint=False)) + list(range(1,10,1))

if __name__ == "__main__":
    for ratio in ratios:
        os.system(f"python3 run.py logs/ca{ratio} 2 {ratio}")