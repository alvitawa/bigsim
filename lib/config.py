import numpy as np

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

cfg = config["DEFAULT"]
IPYTHON_MODE = cfg.getboolean("ipython")
SLIDERS = cfg.getboolean("sliders")
DEFAULT_SAVE = cfg["save"]
WIDTH = int(cfg["width"])
HEIGHT = int(cfg["height"])
CLUSTERING_METHOD = cfg["clustering_method"]

# Maybe parse the config more nice than this, by setting the types and then referring to just cfg["WIDTH"] weetjewel
HEADLESS = cfg.getboolean("headless")

# Processing of parameters
GRID_SIZE = cfg.getfloat("env_size") ** 2 / cfg.getfloat("grid_count")
BOX_SIGHT = np.ceil(cfg.getfloat("max_fish_range") / GRID_SIZE)