import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from capability_analyser import capability_analyser
from robot_distribution import victim_clustering, robot_distribution

from robot import Robot
from victim import Victim
from goal_allocation import make_span_calc, final_allocation


make_span = {'FirstAids': 15, 'DebrisRemover': 30, 'OxygenCylinder': 20,
             'Defuser': 10, 'Manipulator': 45, 'FireExtinguisher': 35}
num_clusters = 5
v1 = Victim(0, [2., 2.], make_span, ['DebrisRemover', 'OxygenCylinder'])
v2 = Victim(1, [0., 13.], make_span, ['Defuser', 'Manipulator'])
v3 = Victim(2, [7., 15.], make_span, ['DebrisRemover', 'FireExtinguisher'])
v4 = Victim(3, [7., 1.], make_span, ['OxygenCylinder', 'Manipulator'])
v5 = Victim(4, [7., 10.], make_span, ['FirstAids', 'DebrisRemover'])
v6 = Victim(5, [6., 12.], make_span, ['Manipulator', 'FireExtinguisher'])
v7 = Victim(6, [5., 7.], make_span, ['FirstAids', 'Manipulator'])
v8 = Victim(7, [9., 19.], make_span, ['FirstAids', 'Defuser'])
v9 = Victim(8, [15., 14.], make_span, ['DebrisRemover', 'Defuser'])
v10 = Victim(9, [19., 11.], make_span, ['FirstAids', 'Manipulator'])

victims = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

r1 = Robot(0, [0., 0.], 0.2, [], num_clusters, [], ['Defuser', 'DebrisRemover'])
r2 = Robot(1, [0., 19.], 0.3, [], num_clusters, [], ['FirstAids', 'OxygenCylinder', 'Manipulator'])
r3 = Robot(2, [19., 0.], 0.4, [], num_clusters, [], ['Manipulator'])
r4 = Robot(3, [19., 19.], 0.3, [], num_clusters, [], ['FirstAids', 'Defuser'])

robots = [r1, r2, r3, r4]

capability_analyser(robots, victims)
print(f'Robot 1 tasks based on capabilities: {r1.tasks}\n'
      f'Robot 2 tasks based on capabilities: {r2.tasks}\n'
      f'Robot 3 tasks based on capabilities: {r3.tasks}\n'
      f'Robot 4 tasks based on capabilities: {r4.tasks}\n')

clusters, clusters_coord = victim_clustering(num_clusters, victims)
victims_new = robot_distribution(robots, victims, clusters, clusters_coord)

make_span_calc(robots, victims)
final_allocation(robots, victims_new)

fig, ax = plt.subplots(1, 1)
for idx, cluster in enumerate(clusters_coord):
    ax.scatter(cluster[0], cluster[1], c="red", marker="^")
    ax.text(cluster[0], cluster[1], f'cluster id: {idx}')
print(clusters_coord, clusters)
for robot in robots:
    print(robot.tasks_init)
for robot in robots:
    print(robot.tasks_final)
for victim in victims:
    print(victim.rescued)
    ax.scatter(victim.pos[0], victim.pos[1], c="blue", marker="s")
    ax.text(victim.pos[0], victim.pos[1], f'victim id: {victim.id}')
# ax.set_xlim(0, 20)
# ax.set_ylim(0, 20)


vor = Voronoi(clusters_coord)
voronoi_plot_2d(vor, ax)
plt.show()
