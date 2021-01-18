import configparser

config = configparser.ConfigParser()
config.read("config.ini")

cfg = config["DEFAULT"]
IPYTHON_MODE = cfg["ipython"] == "1"
SLIDERS = cfg["sliders"] == "1"
DEFAULT_SAVE = cfg["save"]
WIDTH = int(cfg["width"])
HEIGHT = int(cfg["height"])
CLUSTERING_METHOD = cfg["clustering_method"]
