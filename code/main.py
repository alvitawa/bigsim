from boid import *
from population import *
from game import *

import time

# Parameters
boid_count = 100
environment_size = np.array([10, 10])
iterations = 10000

boid_speed = 0.04

# Init population
n_boids = 20
population = Population(boid_count, boid_speed, environment_size)

# Init pygame
screen = init_pygame(resolution=[1080, 720])

# Simulation loop!
for i in range(iterations):
    success = draw_population(population, screen)

    population.iterate(1)

    # pygame.time.delay(100)

    if not success:
        break

# Clean up
exit_pygame()
