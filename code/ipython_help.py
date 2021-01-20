from IPython import embed

import sys
import pdb

# dit zou eig een class moeten zijn alfonso

simulation_ref = None

def exception_catcher(f, *args, **kwargs):
    global exc
    try:
        f(*args, **kwargs)
    except Exception as e:
        exc = sys.exc_info()
        print(e)

def set_sim_ref(simulation): # Todo zoiets implementeren
    global simulation_ref
    simulation_ref = simulation

def debug():
    pdb.post_mortem(exc[2])

def pars():
    global simulation_ref
    return simulation_ref.pars
    
def lp():
    global simulation_ref
    return simulation_ref.load()

def wp():
    global simulation_ref
    return simulation_ref.save()