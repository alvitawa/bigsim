# Simple pygame program
# Import and initialize the pygame library
import pygame
import numpy as np
import math

from data import Population

OCEAN_COLOR = (79, 66, 181)
BOID_COLOR = (249, 166, 2)

# Set up pygame
def init_pygame(resolution=[1080, 720]):
    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    screen = pygame.display.set_mode(resolution)

    clock = pygame.time.Clock()

    global fish
    fish = pygame.image.load("sprites/fish.png")
    fish = pygame.transform.scale(fish, (25, 25))

    return screen, clock

def exit_pygame():
    pygame.quit()

def draw_population(population: Population, screen):
    # Exit nicely
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                population.plot()

    # Fill ocean background
    screen.fill((0, 0, 0))

    scaling = np.array(pygame.display.get_window_size()) / population.env.shape

    for boid in population.population:
        location = tuple((boid[0] * scaling))

        xness = location[0] / pygame.display.get_window_size()[0]
        if math.isnan(xness):
            xness = 0
        
        yness = location[1] / pygame.display.get_window_size()[1]
        if math.isnan(yness):
            yness = 0

        yness = 1
        
        gradient = 0.25 + 0.75 * yness ** 2
        color = (int(249 * gradient), int(166 * gradient), int(2 * gradient))

        # pygame.draw.circle(screen, (wowa, wowb, wowc), location, 5)

        rotation = -np.arccos(boid[1][0]) + 0.5 * np.pi

        if math.isnan(rotation):
            rotation = 0

        draw_triangle(screen, boid[0] * scaling, rotation, color)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

        

    # Update screen
    pygame.display.flip()

    return True


def draw_triangle(surface, position, rotation, color = BOID_COLOR, length = 5, width = 3):
    head_up_down = np.array([[0.5 * length, 0], [-0.5 * length, 0.5 * width], [-0.5 * length, -0.5 * width]])

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    rotated = R.dot(head_up_down.T).T

    rotated += position

    positions = [(int(np.round(a)), int(np.round(b))) for a, b in rotated]

    pygame.draw.polygon(surface, color, positions, width=2)


# Done! Time to quit.
pygame.quit()