# Simple pygame program
# Import and initialize the pygame library
import pygame
from pygame_widgets import Slider, TextBox, Button
from sliders import LabeledSlider
import numpy as np
import math

import data
from data import BoidParameters, Population

OCEAN_COLOR = (49, 36, 131) # (255, 255, 255) 
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
]

# Set up pygame
def init_pygame(boid_parameters, resolution=[1080, 720]):
    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    clock = pygame.time.Clock()

    global fish
    fish = pygame.image.load("sprites/fish.png")
    fish = pygame.transform.scale(fish, (25, 25))

    global sliders
    sliders = []

    for n, par in enumerate(SLIDABLE_PARAMETERS):
        slider = LabeledSlider(
            screen, 10, resolution[1] - 60 - n * 40, par, initial=boid_parameters[par], min=0, max=14
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
        population.boid[par] = slider.get_value()

    # Keyboard presses
    for event in events:
        # Exit nicely
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN:
            # plot
            if event.key == pygame.K_p:
                pass
                # population.plot() TODO REIMPLEMENT THIS
            # toggle rules
            if event.key == pygame.K_1:
                data.do_cohesion = not data.do_cohesion
                print("Cohesion rule is now ", data.do_cohesion)

            if event.key == pygame.K_2:
                data.do_alignment = not data.do_alignment
                print("Alignment rule is now ", data.do_alignment)

            if event.key == pygame.K_3:
                data.do_seperation = not data.do_seperation
                print("Seperation rule is now ", data.do_seperation)
    return False


def clear_screen(screen):
    # Fill ocean background
    screen.fill(OCEAN_COLOR)


def draw_sliders():
    for slider in sliders:
        slider.draw()


def draw_population(population: Population, screen):

    scaling = np.array(pygame.display.get_window_size()) / population.env.shape

    for boid in population.population:
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

        draw_triangle(screen, boid[0] * scaling, rotation, BOID_COLOR)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

    return True


def update_screen():
    # Update screen
    pygame.display.flip()


def draw_triangle(surface, position, rotation, color=BOID_COLOR, length=20, width=8):
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
