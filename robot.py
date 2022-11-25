import numpy as np


class Robot:
    def __init__(self, id, pos, speed, algorithms, num_clusters, competency, capabilities):
        self.id = id
        self.init_pos = pos
        self.pos = pos
        self.speed = speed
        self.w_rc = np.zeros((num_clusters,))
        self.algorithms = algorithms
        self.competency = competency
        self.capabilities = capabilities
        self.tasks = []
        self.tasks_init = []
        self.tasks_init_dist = []
        self.tasks_final = []

    def travel_cost(self):
        X_i = self.tasks_init_dist[0]
        X_f = np.linalg.norm(np.subtract(self.tasks_init[0], self.tasks_init[-1]))
        T_i = X_i / self.speed
        T_f = X_f / self.speed
        return T_f - T_i
