# Simple pygame program
# Import and initialize the pygame library
import pygame
from pygame_widgets import Slider, TextBox, Button
from sliders import LabeledSlider
import numpy as np
import math

import data
from data import Population

OCEAN_COLOR = (255, 255, 255) # (79, 66, 181)
BOID_COLOR = (0, 0, 0) # (249, 166, 2)

# Set up pygame
def init_pygame(resolution=[1080, 720]):
    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    clock = pygame.time.Clock()

    global fish
    fish = pygame.image.load("sprites/fish.png")
    fish = pygame.transform.scale(fish, (25, 25))

    global cohesion_slider, adhesion_slider, seperation_slider
    cohesion_slider = LabeledSlider(screen, 10, resolution[1]-100, "cohesion")
    adhesion_slider = LabeledSlider(screen, 10, resolution[1]-60, "adhesion")
    seperation_slider = LabeledSlider(screen, 10, resolution[1]-20, "seperation")

    return screen, clock

def exit_pygame():
    pygame.quit()

def check_input():
    events = pygame.event.get()

    # Update sliders
    cohesion_slider.update(events)
    adhesion_slider.update(events)
    seperation_slider.update(events)


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
    cohesion_slider.draw()
    adhesion_slider.draw()
    seperation_slider.draw()

def draw_population(population: Population, screen):

    scaling = np.array(pygame.display.get_window_size()) / population.size

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

        if math.isnan(boid[1][0]):
            rotation = 0
        else:
            rotation = -np.arccos(boid[1][0]) + 0.5 * np.pi

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


def draw_triangle(surface, position, rotation, color = BOID_COLOR, length = 10, width = 5):
    head_up_down = np.array([[0.5 * length, 0], [-0.5 * length, 0.5 * width], [-0.5 * length, -0.5 * width]])

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    rotated = R.dot(head_up_down.T).T

    rotated += position

    positions = [(int(np.round(a)), int(np.round(b))) for a, b in rotated]

    pygame.draw.polygon(surface, color, positions, width=0)


# Done! Time to quit.
pygame.quit()