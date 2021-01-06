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
    screen = pygame.display.set_mode(resolution)
    global fish
    fish = pygame.image.load("sprites/fish.png")
    fish = pygame.transform.scale(fish, (25, 25))

    return screen

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

    scaling = np.array(pygame.display.get_window_size()) / population.size

    for boid in population.population:
        location = tuple((boid[0] * scaling))

        xness = location[0] / pygame.display.get_window_size()[0]
        if math.isnan(xness):
            xness = 0
        
        yness = location[1] / pygame.display.get_window_size()[1]
        if math.isnan(yness):
            yness = 0
        
        wowa = int(249 * yness ** 2)
        wowb = int(166 * yness ** 2)
        wowc = int(2 * yness ** 2)

        pygame.draw.circle(screen, (wowa, wowb, wowc), location, 5)

        # print(boid[1][0])

        # rotation = -np.arccos(boid[1][0]) * 180 / np.pi

        # fish_rect = fish.get_rect()
        # fish_rect.center = location

        # screen.blit(pygame.transform.rotate(fish, rotation), fish_rect)

        

    # Update screen
    pygame.display.flip()

    return True


# Done! Time to quit.
pygame.quit()