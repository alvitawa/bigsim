from boid import *

import numpy as np
import matplotlib.pyplot as plt

class Population:
    """
    This class is for all boids.
    """
    
    def __init__(self, n_boids, boid_speed, dim):
        self.boids = []
        for boid in range(n_boids):
            pos = (np.random.rand(2, 1).T * np.array(dim)).T
            boid = Boid(pos, boid_speed)
            self.boids.append(boid)

        self.dim = dim
        
    # Iterates n times
    def iterate(self, n):
        for i in range(n):
            # Update the neighbors
            self.update_neighbors()

            # Move boids
            for boid in self.boids:
                boid.move()

            # Wrap around
            self.wrap_boids()

    def iterate_neighbours(self, n):
        for i in range(n):
            # Move boids
            for boid1 in self.boids:

                # Get Neighbours
                school_neighbors = []
                close_neighbors = []

                # Other boids
                for boid2 in self.boids:
                    if boid1 != boid2:

                        # Calculate distance
                        distance = np.linalg.norm(boid1.position - boid2.position)

                        # Add to school neighbors if close
                        if (distance < boid1.sight_radius):
                            school_neighbors.append(boid2)

                            # Add to social distancing if too close
                            if (distance < boid1.boid_radius):
                                close_neighbors.append(boid2)

                # Move boids
                boid1.move(school_neighbors, close_neighbors)

            # Wrap around
            self.wrap_boids()

    def iterate_boxes(self, n):
        for i in range(n):
            pass

    def wrap_boids(self):
        for boid in self.boids:
            boid.position[0] %= self.dim[0]
            boid.position[1] %= self.dim[1]
    
    # Updates the self.neighbors list
    def update_neighbors(self):
        # First boids
        for boid1 in self.boids:
            # Reset neighbors
            school_neighbors = []
            close_neighbors = []

            # Other boids
            for boid2 in self.boids:
                if boid1 != boid2:

                    # Calculate distance
                    distance = np.linalg.norm(boid1.position - boid2.position)

                    # Add to school neighbors if close
                    if (distance < boid1.sight_radius):
                        school_neighbors.append(boid2)

                        # Add to social distancing if too close
                        if (distance < boid1.boid_radius):
                            close_neighbors.append(boid2)

            boid1.set_neighbors(school_neighbors, close_neighbors)

    # Plots all boids
    def plot(self, ax=None):
        positions = np.hstack([boid.position for boid in self.boids])
        direction = np.hstack([boid.moving_vector for boid in self.boids])

        if not ax:
            fig, ax = plt.subplots()

            ax.set_xlim(0, self.dim[0])
            ax.set_ylim(0, self.dim[1])

            ax.quiver(positions[0], positions[1], direction[0], direction[1])
            plt.show()
        else:
            ax.quiver(positions[0], positions[1], direction[0], direction[1])