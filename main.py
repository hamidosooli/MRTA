import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from capability_analyser import capability_analyser, assistance
from robot_distribution import victim_clustering, robot_distribution

from robot import Robot
from victim import Victim
from goal_allocation import final_allocation, finalizer


make_span = {'FirstAids': 15, 'DebrisRemover': 30, 'OxygenCylinder': 20,
             'Defuser': 10, 'Manipulator': 45, 'FireExtinguisher': 35}
num_clusters = 5
v0 = Victim(0, [2., 2.], make_span, ['DebrisRemover', 'OxygenCylinder'])
v1 = Victim(1, [0., 13.], make_span, ['Defuser', 'Manipulator'])
v2 = Victim(2, [7., 15.], make_span, ['DebrisRemover', 'FireExtinguisher'])
v3 = Victim(3, [7., 1.], make_span, ['OxygenCylinder', 'Manipulator'])
v4 = Victim(4, [7., 10.], make_span, ['FirstAids', 'DebrisRemover'])
v5 = Victim(5, [6., 12.], make_span, ['Manipulator', 'FireExtinguisher'])
v6 = Victim(6, [5., 7.], make_span, ['FirstAids', 'Manipulator'])
v7 = Victim(7, [9., 19.], make_span, ['FirstAids', 'Defuser'])
v8 = Victim(8, [15., 14.], make_span, ['DebrisRemover', 'Defuser'])
v9 = Victim(9, [19., 11.], make_span, ['FirstAids', 'Manipulator'])

victims = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]

r0 = Robot(0, [0., 0.], 0.2, [], num_clusters, [], ['Defuser', 'DebrisRemover'])
r1 = Robot(1, [0., 19.], 0.3, [], num_clusters, [], ['FirstAids', 'OxygenCylinder', 'Manipulator'])
r2 = Robot(2, [19., 0.], 0.4, [], num_clusters, [], ['Manipulator'])
r3 = Robot(3, [19., 19.], 0.3, [], num_clusters, [], ['FirstAids', 'Defuser'])

robots = [r0, r1, r2, r3]
capability_analyser(robots, victims)
human_assistance = assistance(victims)
print(f'Robot 0 tasks based on capabilities: {r0.tasks}\n'
      f'Robot 1 tasks based on capabilities: {r1.tasks}\n'
      f'Robot 2 tasks based on capabilities: {r2.tasks}\n'
      f'Robot 3 tasks based on capabilities: {r3.tasks}\n')

print(f'Robot 0 tasks based on full satisfaction of the capabilities: {r0.tasks_full}\n'
      f'Robot 1 tasks based on full satisfaction of the capabilities: {r1.tasks_full}\n'
      f'Robot 2 tasks based on full satisfaction of the capabilities: {r2.tasks_full}\n'
      f'Robot 3 tasks based on full satisfaction of the capabilities: {r3.tasks_full}\n')

clusters, clusters_coord = victim_clustering(num_clusters, victims)
victims_new = robot_distribution(robots, victims, clusters, clusters_coord)

victims_new = final_allocation(robots, victims, victims_new)
victims_new = finalizer(robots, victims_new, victims)

fig, ax = plt.subplots(1, 1)
for idx, cluster in enumerate(clusters_coord):
    ax.scatter(cluster[0], cluster[1], c="red", marker="^")
    ax.text(cluster[0], cluster[1], f'cluster id: {idx}')
for robot in robots:
    print(f'{robot.id} --> Step 1: {robot.tasks_init}, Step 2: {robot.tasks_final}, Step 3: {robot.tasks_finalized}')

for victim in victims_new:
    print(victim.id, victim.rescued, victim.capabilities, victim.candidates)

for victim in victims:
    print(victim.rescued)
    ax.scatter(victim.pos[0], victim.pos[1], c="blue", marker="s")
    ax.text(victim.pos[0], victim.pos[1], f'victim id: {victim.id}')

vor = Voronoi(clusters_coord)
voronoi_plot_2d(vor, ax)
plt.show()
