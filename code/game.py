# Simple pygame program
# Import and initialize the pygame library
import pygame
from pygame_widgets import Slider, TextBox, Button
from sliders import LabeledSlider
import numpy as np
import math

from sklearn.mixture import GaussianMixture

import data
from data import Simulation

OCEAN_COLOR = (0, 0, 0) # (49, 36, 131) # (255, 255, 255) 
BOID_COLOR = (219, 126, 67) # (0, 0, 0) 

SLIDABLE_PARAMETERS = [
    "speed",
    "agility",
    "separation_weight",
    "separation_range",
    "cohesion_weight",
    "cohesion_range",
    "alignment_weight",
    "alignment_range",
    "obstacle_weight",
    "obstacle_range",
]

def init_globals():
    global GM
    global K

    global COLORS

    K = 5

    # Init K different colors
    COLORS = np.random.choice(range(256), size=3*K).reshape(K, 3)

    GM = GaussianMixture(n_components=K, 
                     max_iter=1000, 
                     tol=1e-4,
                     init_params='random')

    GM.fit(np.random.rand(K, 2))


# Set up pygame
def init_pygame(simulation_pars, resolution=[1080, 720], do_sliders=True):
    init_globals()

    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    clock = pygame.time.Clock()

    global sliders
    sliders = []

    if do_sliders:
        for n, par in enumerate(SLIDABLE_PARAMETERS):
            slider = LabeledSlider(
                screen, 10, resolution[1] - 60 - n * 40, par, initial=simulation_pars[par], min=0, max=14
            )
            sliders.append(slider)

    return screen, clock


def exit_pygame():
    pygame.quit()


def check_input(population):
    events = pygame.event.get()

    # Update sliders
    for par, slider in zip(SLIDABLE_PARAMETERS, sliders):
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
                scaled = pos / np.array(pygame.display.get_window_size()) * population.pars.shape

                new_shape = np.array(population.obstacles.shape)
                new_shape[0] += 1
                population.obstacles = np.append(population.obstacles, scaled).reshape(new_shape)

    return False


def clear_screen(screen):
    # Fill ocean background
    screen.fill(OCEAN_COLOR)


def draw_sliders():
    for slider in sliders:
        slider.draw()

def draw_number(screen, number):
        '''Displays a number on the screen'''
        font = pygame.font.SysFont('arial', 50)
        text = font.render(str(number), True, (0, 0, 0))
        screen.blit(text, (0,0))
        # pygame.display.update()

def positions_to_colors(positions):
    global GM
    global K
    global COLORS

    # New GMM based on GMM of last iteration
    GM = GaussianMixture(n_components=K, 
                        max_iter=1000, 
                        tol=1e-4,
                        means_init=GM.means_,
                        weights_init=GM.weights_,)
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
        location = tuple((boid[0] * scaling))

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

        draw_triangle(screen, boid[0] * scaling, rotation, boid_color)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

    for obstacle in population.obstacles:
        location = tuple((obstacle[0] * scaling))

        pygame.draw.circle(screen, (245, 40, 40), location, 6)


    return True


def update_screen():
    # Update screen
    pygame.display.flip()



def draw_triangle(surface, position, rotation, color=BOID_COLOR, length=30, width=15):
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
