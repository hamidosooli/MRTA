import numpy as np


class Victim:
    def __init__(self, id, pos, make_span, capabilities):
        self.id = id
        self.pos = pos
        self.capabilities = capabilities
        self.cluster_id = np.nan
        self.cluster_dist = np.nan
        self.rescued = False
        self.make_span = make_span

