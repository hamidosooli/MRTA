import numpy as np


class Victim:
    def __init__(self, id, pos, make_span, requirements):
        self.id = id
        self.pos = pos
        self.capabilities = requirements  # Victim requirements
        self.rem_req = []  # Victim remaining requirements
        self.candidates = [[] for _ in range(len(self.capabilities))]
        self.cluster_id = np.nan
        self.cluster_dist = np.nan
        self.rescued = np.zeros_like(requirements, dtype=bool)
        self.make_span = make_span
        self.health_stt = np.random.choice([1, .6, .3])  # 1: high health, .6: low health, .3:critical health
