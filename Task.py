class task:
    def __init__(self, id, pos, area, hazard, capabilities):
        self.id = id
        self.pos = pos
        self.area = area  # in SI unit (m^2)
        self.hazard = hazard
        self.capabilities = capabilities