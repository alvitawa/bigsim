# Simple pygame program
# Import and initialize the pygame library
import pygame
from pygame_widgets import Slider, TextBox, Button
from sliders import LabeledSlider
import numpy as np
import math

from sklearn.mixture import GaussianMixture

import warnings
from sklearn.exceptions import ConvergenceWarning

import data
from data import Simulation

OCEAN_COLOR = (0, 0, 0) # (49, 36, 131) # (255, 255, 255) 
BOID_COLOR = (219, 126, 67) # (0, 0, 0) 

SLIDABLE_PARAMETERS = [
#   Name                    Max Value
    ("speed",               1),
    ("agility",             1),
    ("separation_weight",   15),
    ("separation_range",    1),
    ("cohesion_weight",     15),
    ("cohesion_range",      1),
    ("alignment_weight",    15),
    ("alignment_range",     1),
    ("obstacle_weight",     15),
    ("obstacle_range",      1),
    ("shark_weight",        15),
    ("shark_range",         1),
    ("shark_speed",         1),
    ("shark_agility",       1),
]

def init_globals():
    global GM
    global K

    global COLORS
    global MENU
    MENU = False

    K = 5

    # Init K different colors
    COLORS = np.random.choice(range(256), size=3*K).reshape(K, 3)

    GM = GaussianMixture(n_components=K, 
                    max_iter=1000, 
                    tol=1e-4,
                    init_params='random',
                    verbose=0)

    # No convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        GM.fit(np.random.rand(K, 2))


# Set up pygame
def init_pygame(simulation_pars, resolution=[1080, 720], do_sliders=True):
    init_globals()

    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    global BUTTON_DATA
    BUTTON_DATA = [
#   Button rectangle                            function
        [[0, screen.get_height()-20, 20, 20],   toggle_menu],
    ]

    clock = pygame.time.Clock()

    global sliders
    sliders = []

    if do_sliders:
        for n, (par, max_value) in enumerate(SLIDABLE_PARAMETERS):
            print(par, type(max_value))
            slider = LabeledSlider(
                screen, 10, resolution[1] - 60 - n * 40, par, initial=simulation_pars[par], min=0, max=max_value
            )
            sliders.append(slider)

    return screen, clock


def exit_pygame():
    pygame.quit()

def load_pars(population):
    pars = population.load()
    for (par, _), slider in zip(SLIDABLE_PARAMETERS, sliders):
        slider.set_value(pars[par])

def check_input(population):
    events = pygame.event.get()

    # Update sliders
    for (par, _), slider in zip(SLIDABLE_PARAMETERS, sliders):
        slider.update(events)
        population.pars[par] = slider.get_value()

    # Keyboard presses
    for event in events:
        # Exit nicely
        if event.type == pygame.QUIT:
            return True

        # example
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pass

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # left click
                pos = np.array(pygame.mouse.get_pos())

                # Check all buttons
                for rectangle, button_function in BUTTON_DATA:
                    if is_in_rect(pos, rectangle):
                        button_function()

                scaled = pos / np.array(pygame.display.get_window_size()) * population.pars.shape

                new_shape = np.array(population.obstacles.shape)
                new_shape[0] += 1
                population.obstacles = np.append(population.obstacles, scaled).reshape(new_shape)

    return False


def clear_screen(screen):
    # Fill ocean background
    screen.fill(OCEAN_COLOR)

def is_in_rect(coordinates, rect):
    in_x = (rect[0] <= coordinates[0] <= rect[0]+rect[2])
    in_y = (rect[1] <= coordinates[1] <= rect[3]+rect[1])
    return in_x and in_y

def draw_sliders():
    if MENU:
        for slider in sliders:
            slider.draw()

# BUTTONS
def toggle_menu():
    global MENU
    MENU = not MENU

def draw_button(surface, rectangle, color=[255, 255, 255], hover=False):
    if hover: 
        # Light is same color minus 10
        color_light = np.max(np.vstack([[0, 0, 0], np.array(color)-100]), axis=0) # This should be easier ?

        pygame.draw.rect(surface,color_light,rectangle) 
    else: 
        pygame.draw.rect(surface,color,rectangle) 

def draw_buttons(screen):
    global BUTTON_DATA
    for rectangle, _ in BUTTON_DATA:

        mouse = pygame.mouse.get_pos() 
        if ( is_in_rect(mouse, rectangle) ): 
            hover = True
        else:
            hover = False

        draw_button(screen, rectangle, hover=hover)

def draw_number(screen, number):
        '''Displays a fps number on the screen'''
        font = pygame.font.SysFont('arial', 50)
        text = font.render(str(number), True, np.abs(np.array(OCEAN_COLOR)-255))
        screen.blit(text, (0,0))
        # pygame.display.update()

def positions_to_colors(positions):
    global GM
    global K
    global COLORS

    # New GMM based on GMM of last iteration
    GM = GaussianMixture(n_components=K, 
                        max_iter=10, 
                        tol=1e-4,
                        means_init=GM.means_,
                        weights_init=GM.weights_,
                        verbose=0)
    
    # No convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        GM.fit(positions)

    probs = GM.predict_proba(positions)

    # Convert probabilities to colors
    return np.sum(probs[:,:,None]*COLORS[None,:,:], axis=1).astype(int)

def draw_population(population: Simulation, screen):
    scaling = np.array(pygame.display.get_window_size()) / population.pars.shape

    # Coloring with GMM
    positions = population.population[:,0,:]
    colors = positions_to_colors(positions)

    for boid, boid_color in zip(population.population, colors):
        # xness = location[0] / pygame.display.get_window_size()[0]
        # if math.isnan(xness):
        #     xness = 0

        # yness = location[1] / pygame.display.get_window_size()[1]
        # if math.isnan(yness):
        #     yness = 0

        # gradient = 0.25 + 0.75 * yness ** 2
        # color = (int(249 * gradient), int(166 * gradient), int(2 * gradient))

        # pygame.draw.circle(screen, (wowa, wowb, wowc), location, 5)

        # fun use unit_dir instead of boid[1]
        # mouse = np.array(location) - np.array(pygame.mouse.get_pos())
        # unit_dir = (mouse) / np.linalg.norm(mouse)

        if boid[1][1] >= 0:
            rotation = np.arccos(boid[1][0])
        else:
            rotation = -np.arccos(boid[1][0])

        draw_fish(screen, boid[0] * scaling, rotation, boid_color, 15, 7)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

    for obstacle in population.obstacles:
        location = tuple((obstacle[0] * scaling))

        pygame.draw.circle(screen, (245, 40, 40), location, 6)

    for shark in population.sharks:
        if shark[1][1] >= 0:
            rotation = np.arccos(shark[1][0])
        else:
            rotation = -np.arccos(shark[1][0])

        draw_fish(screen, shark[0] * scaling, rotation, (117,57,57), 120, 40)

    return True


def update_screen():
    # Update screen
    pygame.display.flip()



def draw_fish(surface, position, rotation, color=BOID_COLOR, length=30, width=15):
    head_up_down = np.array(
        [[0.5 * length, 0],
        [0.25 * length, 0.5 * width],
        [-0.25 * length, 0.1 * width],
        [-0.5 * length, 0.5 * width],
        [-0.5 * length, -0.5 * width],
        [-0.25 * length, -0.1 * width],
        [0.25 * length, -0.5 * width]
        ]
    )

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    rotated = R.dot(head_up_down.T).T

    rotated += position

    positions = [(int(np.round(a)), int(np.round(b))) for a, b in rotated]

    pygame.draw.polygon(surface, color, positions, width=0)


# Done! Time to quit.
pygame.quit()
