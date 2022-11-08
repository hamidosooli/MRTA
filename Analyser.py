from Agent import agent
from Task import task


# Define agent's specifications (id, speed, sensors, algorithms, competency)
robot1 = agent(0, [5, 5], 3,
               ['LIDAR'],
               ['RANDOM_WALK', 'ANT_COLONY'],
               ['RECEIVE_INFO'], ['communicate-data', 'check-temperature', 'refuel'])

robot2 = agent(1, [10, 15], 2,
               ['STEREO_CAM'],
               ['LINE_BY_LINE', 'LEVY_WALK'],
               ['SEND_INFO'], ['navigation', 'valve-inspection', 'check-pressure'])

robot3 = agent(2, [19, 15], 2,
               ['LIDAR', 'STEREO_CAM', 'FUSE'],
               ['ANT_COLONY'],
               ['SEND_INFO', 'RECEIVE_INFO'], ['turn-valve', 'observation', 'take-image'])

# Define tasks specifications (id, area, hazard)
room1 = task(0, [0, 0], 4*4, ['FLAME', 'HEAVY_SMOKE'], ['valve-inspection', 'check-pressure'])
room2 = task(1, [0, 19], 5*5, ['HEAVY_SMOKE', 'SPLASH'], ['check-temperature', 'turn-valve'])
room3 = task(2, [19, 0], 5*6, ['FLAME', 'EXPLOSIVES'], ['refuel', 'communicate-data'])
room4 = task(3, [19, 19], 8*4, ['SPLASH'], ['navigation', 'observation', 'take-image'])

cap_set = ['navigation', 'turn-valve', 'observation',
           'take-image', 'refuel', 'communicate-data',
           'check-temperature', 'valve-inspection', 'check-pressure']


def capability_analyser(capability_set, agents, tasks):

    for agent in agents:
        for capability in capability_set:
            for task in tasks:
                if capability in agent.capabilities and capability in task.capabilities:
                    if task.id not in agent.tasks:
                        agent.tasks.append(task.id)
        print(f'Agent {agent.id} will do tasks {agent.tasks}')


def sensor_task(agents, tasks):

    num_agents = len(agents)
    num_tasks = len(tasks)

    assignment = []
    for i in range(num_agents):
        for j in range(num_tasks):
            for sns in agents[i].sensors:
                for hzrd in tasks[j].hazard:
                    if sns == 'FUSE' and (hzrd == 'SPLASH' or hzrd == 'HEAVY_SMOKE'):
                        if [j, i] not in assignment:
                            assignment.append([j, i])   # [task, agent]
                    elif sns == 'LIDAR' and hzrd == 'EXPLOSIVES':
                        if [j, i] not in assignment:
                            assignment.append([j, i])
                    elif sns == 'STEREO_CAM' and hzrd == 'FLAME':
                        if [j, i] not in assignment:
                            assignment.append([j, i])
    return assignment


assgn = sensor_task([robot1, robot2, robot3], [room1, room2, room3, room4])

capability_analyser(cap_set, [robot1, robot2, robot3], [room1, room2, room3, room4])

pass