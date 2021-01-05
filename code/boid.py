import numpy as np
import matplotlib.pyplot as plt

DIMENSIONS = 2

def norm_v(vector):
    return vector/np.linalg.norm(vector)

class Boid:
    """
    This class is for one Boid.
    """
    # # physique
    # position = np.array([])
    # moving_vector = np.array([])

    # # parameters
    # speed = 0
    # sight_radius = 0
    # pivot_speed = 0

    # # other
    # neighbors = []

    
    def __init__(self, 
                 start_position, 
                 start_speed=0.05, 
                 start_sight_radius=0.3, 
                 start_boid_radius=0.1,
                 pivot_speed=0.5,
                 ):
        self.position = start_position
        self.moving_vector = norm_v(np.random.rand(DIMENSIONS, 1))
        
        self.speed = start_speed
        self.sight_radius = start_sight_radius
        self.boid_radius = start_boid_radius
        self.pivot_speed = pivot_speed

        self.school_neighbors = []
        self.close_neighbors = []
    
    # Move one boid for one iteration
    def move(self):

        # Change movement vector
        if (len(self.school_neighbors) > 0):
            self.pivot()

        # Normalize moving vector
        self.moving_vector = norm_v(self.moving_vector)
        # Apply movement vector
        movement = (self.moving_vector) * self.speed
        self.position = self.position + movement

    # Updates neighbors
    def set_neighbors(self, school_neighbors, close_neighbors):
        self.school_neighbors = school_neighbors
        self.close_neighbors = close_neighbors

    # Changes the direction
    def pivot(self):
        # Cohesion rule: A boid moves toward the mean position of its "closest neighbors." 
        # Get neighbors
        # Calculate mean
        school_neighbors_mean_pos = np.mean([boid.position for boid in self.school_neighbors])

        # Calculate vector to mean (Cohesion vector)
        coh_vector = school_neighbors_mean_pos - self.position

        # Alignment rule: A boid heads in the mean direction to which its "closest neighbors" head.
        # Get mean vector of closest neighbors vector (Alignment vector)
        ali_vector = np.mean([boid.moving_vector for boid in self.school_neighbors])

        # Separation rule: A boid does not get closer than some minimum distance to any neighbor.
        if (len(self.close_neighbors) > 0):
            close_neighbors_mean_pos = np.mean([boid.position for boid in self.close_neighbors])
            sep_vector = norm_v(self.position - close_neighbors_mean_pos)
        else:
            sep_vector = np.zeros((DIMENSIONS , 1))

        # Combine vectors to one vector
        coh_ali = norm_v(coh_vector) + norm_v(ali_vector) + sep_vector

        # Update moving vector
        self.moving_vector = self.moving_vector + coh_ali * self.pivot_speed

        pass