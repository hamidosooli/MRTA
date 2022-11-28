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

        self.make_span = []
        self.abort_time = 0.0
        self.travel_time = 0.0

    def travel_cost(self):
        X_i = np.linalg.norm(np.subtract(self.tasks_init_dist[0], self.init_pos))
        X_f = np.linalg.norm(np.subtract(self.tasks_init_dist[-1], self.tasks_init_dist[0]))
        T_i = X_i / self.speed
        T_f = X_f / self.speed
        self.travel_time = T_f - T_i
