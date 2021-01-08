from boid import *
from data import *
from population import OldPopulation

import time


boid_count = 300
environment_size = np.array([10, 10])
iterations = 1000

boid_speed = 0.05
rotation_rate = 0.95


grid_size = [1.0, 1.0]

population = Population(boid_count, environment_size, boid_speed, grid_size, rotation_rate)
# population = OldPopulation(boid_count, boid_speed, environment_size)

start = time.perf_counter()
population.iterate(iterations)
end = time.perf_counter()

print(end - start)