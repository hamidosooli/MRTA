# Agent specifications
class agent:
    def __init__(self, id, pos, speed, sensors, algorithms, competency, capabilities):
        '''
        :param id: given number to the agent
        :param speed: maximum speed of the agent
        :param sensors: character set indicting the name of the agent's available sensors
        :param algorithms: character set indicating the name of the available search algorithms
        :param competency: set of the agent's competency profile
        '''
        self.id = id
        self.pos = pos
        self.speed = speed
        self.sensors = sensors
        self.algorithms = algorithms
        self.competency = competency
        self.capabilities = capabilities
        self.tasks = []

    