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

def cluster_LC(positions):
    global COLORS

    assignments = GM.fit(positions)
    return COLORS[assignments]

def get_method(simulation):
    if (len(simulation.sharks) and CLUSTERING_METHOD=="LARS_CLUSTERING"):
        return "GMM"
    else: 
        return CLUSTERING_METHOD


def positions_to_colors(positions):
    global method

    if method == "GMM":
        labels = cluster_GMM(positions)
    elif method == "DBSCAN":
        labels = cluster_DBSCAN(positions)
    elif method == "LARS_CLUSTERING":
        labels = cluster_LC(positions)
    else: # DEFAULT
        labels = cluster_GMM(positions)

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
        if init_points:
            self.leaders = init_points
        else:
            self.leaders = [np.random.randint(0, len(data))]
            
        self.threshold=threshold
        self.clusters = []
        self.current_leader_index = 0
    
    def leader_to_flock(self, leader, data):
        flock = points_around(leader, data, threshold=self.threshold)
        return flock
        
    def get_next_leader(self, remaining, data):
        """
        remaining: remaining fish
        data: all fish

        return: index of next leader in data
        """
        # Get leader from saved leaders
        if self.current_leader_index < len(self.leaders):
            leader_id = self.leaders[self.current_leader_index]
            self.current_leader_index += 1
            
            # if leader in the remaining data
            # while not data[leader_id] in remaining:
            #     self.leaders.remove(leader_id)
            #     leader_id = self.leaders[self.current_leader_index]
            
            return leader_id
        
        # Make new leader
        else:
            new_leader = remaining[np.random.randint(0, len(remaining))]
            
            # To index
            leader_id = np.where(data == new_leader)[0][0]
            
            self.leaders.append(leader_id)
            self.current_leader_index += 1
            
            return leader_id
        
    def fit(self, data):
        
        cluster_assignment = np.array([-1]*len(data))
        current_cluster = 0
            
        while (-1 in cluster_assignment):
            
            # Get unassigned points
            remaining = data[cluster_assignment == -1]
            
            # Get a leader
            leader_id = self.get_next_leader(remaining, data)
            leader = data[leader_id]
            
            # Find the flock of this leader
            flock = self.leader_to_flock(np.array([leader]), data)
            flock_points = data[flock]
            
            # Update assignments
            cluster_assignment[flock] = current_cluster
            current_cluster += 1
            
            if current_cluster == 15:
                break
            
        # RESET
        self.current_leader_index = 0
        
        return cluster_assignment
    