import numpy as np

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
    