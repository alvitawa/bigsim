import numpy as np
import os

ratios = np.linspace(0, 1, 10, endpoint=True)

# Personalize this
python_location = "C:/Users/Lodewijk/AppData/Local/Programs/Python/Python38-32/python.exe"

if __name__ == "__main__":
    for ratio in ratios:
        os.system(python_location + f" run.py logs/coh{ratio} 4 {ratio}")