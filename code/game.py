# Simple pygame program
# Import and initialize the pygame library
import pygame
from pygame_widgets import Slider, TextBox, Button
from sliders import LabeledSlider
import numpy as np
import math

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

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
    ("separation_range",    3),
    ("cohesion_weight",     15),
    ("cohesion_range",      3),
    ("alignment_weight",    15),
    ("alignment_range",     3),
    ("obstacle_weight",     15),
    ("obstacle_range",      3),
    ("shark_weight",        15),
    ("shark_range",         3),
    ("shark_speed",         1),
    ("shark_agility",       1),
]

def init_globals(sim):
    global simulation

    simulation = sim

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
def init_pygame(simulation, resolution=[1080, 720], do_sliders=True):
    init_globals(simulation)

    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    button_data = [
#       Button rectangle                               function                     TEXT
        [[0, screen.get_height()-60, 60, 60],          toggle_menu,                 "MENU"],
        [[screen.get_width()-130, screen.get_height() - 60, 60, 60],   save,        "SAVE"],
        [[screen.get_width()-60, screen.get_height() - 60, 60, 60],    load,        "LOAD"]
    ]

    global BUTTONS
    BUTTONS = []
    for rect, func, text in button_data:
        button = Button(screen, rect[0], rect[1], rect[2], rect[3],
                        text=text,
                        onClick=func)
        BUTTONS.append(button)

    clock = pygame.time.Clock()

    global sliders
    sliders = []

    if do_sliders:
        for n, (par, max_value) in enumerate(SLIDABLE_PARAMETERS):
            print(par, type(max_value))
            slider = LabeledSlider(
                screen, 10, resolution[1] - 60 - n * 40, par, initial=simulation.pars[par], min=0, max=max_value
            )
            sliders.append(slider)

    return screen, clock


def exit_pygame():
    pygame.quit()

def save():
    global simulation
    simulation.save()

def load():
    global simulation
    pars = simulation.load()
    for (par, _), slider in zip(SLIDABLE_PARAMETERS, sliders):
        slider.set_value(pars[par])

def check_input():
    global simulation

    events = pygame.event.get()

    # Button interaction
    for button in BUTTONS:
        button.listen(events)

    # Update sliders
    for (par, _), slider in zip(SLIDABLE_PARAMETERS, sliders):
        slider.update(events)
        simulation.pars[par] = slider.get_value()

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
            if event.button == 3: # right click = select fish
                pos = np.array(pygame.mouse.get_pos())
                scaled = pos / np.array(pygame.display.get_window_size()) * simulation.pars.shape

                fish_rel = simulation.population[:, 0, :] - scaled
                distances = np.sqrt(np.power(fish_rel, 2).sum(axis=-1))

                mindin = np.min(distances)

                if mindin < 1:
                    data.selected_index = np.argmin(distances)
                else:
                    data.selected_index = None

            if event.button == 2: # middle click place obstacle
                pos = np.array(pygame.mouse.get_pos())
                scaled = pos / np.array(pygame.display.get_window_size()) * simulation.pars.shape

                new_shape = np.array(simulation.obstacles.shape)
                new_shape[0] += 1
                simulation.obstacles = np.append(simulation.obstacles, scaled).reshape(new_shape)

    return False


def clear_screen(screen):
    # Fill ocean background
    screen.fill(OCEAN_COLOR)

def draw_sliders():
    if MENU:
        for slider in sliders:
            slider.draw()

# BUTTONS
def toggle_menu():
    global MENU
    MENU = not MENU

def draw_buttons():
    global BUTTONS
    for button in BUTTONS:
        # Always draw menu button
        if button.string == "MENU":
            button.draw()
        # Draw other buttons with menu
        elif (MENU):
            button.draw()
            
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
    # GM = GaussianMixture(n_components=K, 
    #                     max_iter=10, 
    #                     tol=1e-4,
    #                     means_init=GM.means_,
    #                     weights_init=GM.weights_,
    #                     verbose=0)
    
    # No convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # GM.fit(positions)
        clustering = DBSCAN(eps=3, min_samples=2).fit(positions)

    # probs = GM.predict_proba(positions)
    probs = clustering.labels_
    return COLORS[probs]

    # Convert probabilities to colors
    return np.sum(probs[:,:,None]*COLORS[None,:,:], axis=1).astype(int)

def debug_draw(screen):
    global simulation
    
    if data.selected_index == None:
        return
    selected_fish = simulation.population[data.selected_index]
    scaling = np.array(pygame.display.get_window_size()) / simulation.pars.shape

    location = selected_fish[0] * scaling

    OBSTACLE_COLOR = (255, 0, 0)
    SEPERATION_COLOR = (255, 0, 255)
    COHESION_COLOR = (0, 255, 0)
    ALIGNMENT_COLOR = (0, 128, 128)
    SHARK_COLOR = (200, 200, 200)

    # draw ranges
    pygame.draw.circle(screen, OBSTACLE_COLOR, tuple(location), int(simulation.pars.obstacle_range * scaling[0]), width=1)
    pygame.draw.circle(screen, SHARK_COLOR, tuple(location), int(simulation.pars.shark_range * scaling[0]), width=1)

    pygame.draw.circle(screen, SEPERATION_COLOR, tuple(location), int(simulation.pars.separation_range * scaling[0]), width=1)
    pygame.draw.circle(screen, COHESION_COLOR, tuple(location), int(simulation.pars.cohesion_range * scaling[0]), width=1)
    pygame.draw.circle(screen, ALIGNMENT_COLOR, tuple(location), int(simulation.pars.alignment_range * scaling[0]), width=1)

    # get vectors
    assigned_box = selected_fish[0] // simulation.grid_size
    grid_coordinates = simulation.population[:, 0, :] // simulation.grid_size
    outer_idx = (np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= simulation.box_sight_radius)
    cohesion, seperation, alignment, obstacle, wall, shark = data.fish_move_vectors(np.array([selected_fish]), simulation.population[outer_idx], simulation.obstacles, simulation.sharks, simulation.pars)

    pygame.draw.line(screen, OBSTACLE_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + obstacle) * scaling))
    pygame.draw.line(screen, SHARK_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + shark) * scaling))
    
    pygame.draw.line(screen, SEPERATION_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + seperation) * scaling))
    pygame.draw.line(screen, COHESION_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + cohesion) * scaling))
    pygame.draw.line(screen, ALIGNMENT_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + alignment) * scaling))




def draw_population(screen):
    global simulation

    scaling = np.array(pygame.display.get_window_size()) / simulation.pars.shape

    # Coloring with GMM
    positions = simulation.population[:,0,:]
    colors = positions_to_colors(positions)

    for boid, boid_color in zip(simulation.population, colors):
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

        draw_fish(screen, boid[0] * scaling, rotation, boid_color, 7, 4)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

    for obstacle in simulation.obstacles:
        location = tuple((obstacle[0] * scaling))

        pygame.draw.circle(screen, (245, 40, 40), location, 6)

    for shark in simulation.sharks:
        if shark[1][1] >= 0:
            rotation = np.arccos(shark[1][0])
        else:
            rotation = -np.arccos(shark[1][0])

        draw_shark(screen, shark[0] * scaling, rotation, (192,192,192), 40, 40)

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

def draw_shark(surface, position, rotation, color, length, width):
    head_up_down = np.array(
        [[1.2 * length, -0.1 * width],
        [0.7 * length, 0 * width],
        [1.2 * length, 0.1 * width],
        [0.6 * length, 0.35 * width],
        [0.45 * length, 0.6 * width],
        [0.4 * length, 0.35 * width],
        [-0.25 * length, 0.1 * width],
        [-0.4 * length, 0.5 * width],
        [-0.5 * length, -0.5 * width],
        [-0.25 * length, -0.1 * width],
        [0.25 * length, -0.35 * width]
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
