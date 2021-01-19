import numpy as np


import warnings

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from sklearn.exceptions import ConvergenceWarning

from config import CLUSTERING_METHOD

global method

def init_gmm():
    global GM
    global K
    global COLORS

    K = 5
    COLORS = np.random.choice(range(256), size=3*K).reshape(K, 3)

    GM = GaussianMixture(n_components=K, 
                    max_iter=1000, 
                    tol=1e-4,
                    init_params='random',
                    verbose=0)

    # # No convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        GM.fit(np.random.rand(K, 2))
    pass

def init_dbscan():
    global COLORS

    n_colors = 15
    COLORS = np.random.choice(range(256), size=3*n_colors).reshape(n_colors, 3)
    pass

def init_lc(population):
    global GM
    global COLORS
    global simulation

    n_colors = 15
    COLORS = np.random.choice(range(256), size=3*n_colors).reshape(n_colors, 3)
    GM = LarsClustering(population)
    pass

def init_clustering(simulation):
    global method

    method = get_method(simulation)

    if method == "GMM":
        init_gmm()
    elif method == "DBSCAN":
        init_dbscan()
    elif method == "LARS_CLUSTERING":
        init_lc(simulation.population)
    else: # DEFAULT
        init_gmm()

def cluster_GMM(positions):
    global COLORS

    global GM
    global K

    # New GMM based on GMM of last iteration
    GM = GaussianMixture(n_components=K, 
                        max_iter=10, 
                        tol=1e-4,
                        means_init=GM.means_,
                        weights_init=GM.weights_,
                        verbose=0)

    # No convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        GM.fit(positions)

    probs = GM.predict_proba(positions)

    # Convert probabilities to colors
    return np.sum(probs[:,:,None]*COLORS[None,:,:], axis=1).astype(int)

def cluster_DBSCAN(positions):
    global COLORS

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clustering = DBSCAN(eps=3, min_samples=2).fit(positions)
        
    probs = clustering.labels_
    return COLORS[probs]

def cluster_LC(simulation):
    global COLORS

    assignments = GM.fit(simulation)
    return COLORS[assignments]

def get_method(simulation):
    # if (len(simulation.sharks) and CLUSTERING_METHOD=="LARS_CLUSTERING"):
    #     return "GMM"
    # else: 
    return CLUSTERING_METHOD


def positions_to_colors(simulation):
    global method

    if method == "GMM":
        labels = cluster_GMM(simulation.population[:, 0])
    elif method == "DBSCAN":
        labels = cluster_DBSCAN(simulation.population[:, 0])
    elif method == "LARS_CLUSTERING":
        labels = cluster_LC(simulation)
    else: # DEFAULT
        labels = cluster_GMM(simulation.population[:, 0])

    return labels
    

def points_around(some_points, all_points, threshold):
    close_id = np.any(np.linalg.norm(all_points[:, None, :] - some_points[None, :, :],
                                  axis = -1) < threshold,
                   axis=1)
    close = all_points[close_id]


    if close.size == some_points.size:
        return close_id

    return points_around(close, all_points, threshold)

class LarsClustering:
    
    def __init__(self, data, threshold=0.4, init_points=None):
        
        # LEADERS IS AN ARRAY OF INDICES
        # if init_points:
        #     self.leaders = init_points
        # else:
        #     self.leaders = [np.random.randint(0, len(data))]
        self.leader = []
            
        self.threshold=threshold
        self.cluster_assignment = []
        self.clusters = []
        self.current_leader_index = 0
        self.max_clusters = 15
    
    def leader_to_flock(self, leader, data):
        flock = points_around(leader, data, threshold=self.threshold)
        return flock
    
    def random_leader(self):
        return np.where(self.cluster_assignment==-1)[0][0]
        
    def get_next_leader(self):
        """
        remaining: remaining fish
        data: all fish

        return: index of next leader in data
        """
        
        # First yield leader in leaders list of simulation
        for leader in self.leaders:
            if leader:
                yield leader

        # Make new leaders
        while True:
            yield self.random_leader()
        
    def fit(self, sim):
        positions = sim.population[:, 0]

        self.cluster_assignment = np.array([-1]*len(positions))
        current_cluster = 0

        # Set the leader variable
        self.leaders = sim.get_leaders()
        new_leaders = []

        # Current leaders
        leaders = self.get_next_leader()
            
        while (-1 in self.cluster_assignment):
            
            # Get unassigned points
            remaining = positions[self.cluster_assignment == -1]
            
            # Get a leader
            leader_id = next(leaders)
            # Check if the leader does not yet have a flock
            if self.cluster_assignment[leader_id] != -1:
                continue

            new_leaders.append(leader_id)
            leader_position = positions[leader_id]
            
            # Find the flock of this leader
            flock = self.leader_to_flock(np.array([leader_position]), positions)
            
            # Update assignments
            self.cluster_assignment[flock] = current_cluster
            current_cluster += 1
            
            # Break with max clusters
            if current_cluster == self.max_clusters:
                break
            
        # RESET
        self.current_leader_index = 0
        sim.update_leaders(new_leaders)
        
        return self.cluster_assignment
    