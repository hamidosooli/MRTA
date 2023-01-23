import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import h5py
from capability_analyser import capability_analyser, assistance
from robot_distribution import task_clustering, robot_distribution

from robot import Robot
from victim import Victim
from goal_allocation import final_allocation, finalizer


make_span = {'FirstAids': 15, 'DebrisRemover': 30, 'OxygenCylinder': 20,
             'Defuser': 10, 'Manipulator': 45, 'FireExtinguisher': 35}
num_clusters = 6
# v0 = Victim(0, [2., 2.], make_span, ['DebrisRemover', 'OxygenCylinder'])
# v1 = Victim(1, [0., 13.], make_span, ['Defuser', 'Manipulator'])
# v2 = Victim(2, [7., 15.], make_span, ['DebrisRemover', 'FireExtinguisher'])
# v3 = Victim(3, [7., 1.], make_span, ['OxygenCylinder', 'Manipulator'])
# v4 = Victim(4, [7., 10.], make_span, ['FirstAids', 'DebrisRemover'])
# v5 = Victim(5, [6., 12.], make_span, ['Manipulator', 'FireExtinguisher'])
# v6 = Victim(6, [5., 7.], make_span, ['FirstAids', 'Manipulator'])
# v7 = Victim(7, [9., 19.], make_span, ['FirstAids', 'Defuser'])
# v8 = Victim(8, [15., 14.], make_span, ['DebrisRemover', 'Defuser'])
# v9 = Victim(9, [19., 11.], make_span, ['FirstAids', 'Manipulator'])

exp_name = 'FindSurvivors'

plt.rcParams.update({'font.size': 22})
file_name = f'../Multi-Agent-Search-and-Rescue/multi_agent_Q_learning_{exp_name}.hdf5'

env_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
num_rows, num_cols = np.shape(env_map)
ox, oy = np.where(env_map == 1)

victims = []
with h5py.File(file_name, 'r') as f:

    for idx in range(f['victims_num'][0]):

        victims.append(Victim(idx, [float(f[f'victim{idx}_trajectory'][-1][0]),
                                    float(f[f'victim{idx}_trajectory'][-1][1])], make_span,
                              np.asarray(f[f'victims_requirement'])[idx].tolist()))

        # victims[idx].pos = [float(coord) for coord in victims[idx].pos]

        for ind, cap in enumerate(victims[idx].capabilities):
            victims[idx].capabilities[ind] = cap.decode()

# victims = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]


r0 = Robot(0, [0, 9], 0.2, [], num_clusters, [], ['Defuser', 'DebrisRemover'])
r1 = Robot(1, [0, 10],  0.3, [], num_clusters, [], ['FirstAids', 'OxygenCylinder', 'Manipulator'])
r2 = Robot(2, [19, 9],  0.4, [], num_clusters, [], ['Manipulator'])
r3 = Robot(3, [19, 10], 0.3, [], num_clusters, [], ['FirstAids', 'Defuser'])

robots = [r0, r1, r2, r3]
starts = []
for robot in robots:
    starts.append(robot.pos)

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

clusters, clusters_coord = task_clustering(num_clusters, victims, ox, oy)
victims_new = robot_distribution(robots, victims, clusters, clusters_coord)
travel2clusters = []
for robot in robots:
    travel2clusters.append(robot.pos)
print(starts, travel2clusters)
victims_new = final_allocation(robots, victims, victims_new)
victims_new = finalizer(robots, victims_new, victims, ox, oy)

fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.rcParams.update({'font.size': 50})
for idx, cluster in enumerate(clusters_coord):
    ax.scatter(cluster[0], cluster[1], c="red", marker="^")
    ax.text(cluster[0], cluster[1], f'C{idx}')

with h5py.File(f'MRTA.hdf5', 'w') as f:
    f.create_dataset(f'RS_size', data=len(robots))
    f.create_dataset(f'Victims_size', data=len(victims))
    f.create_dataset(f'RS_starts', data=starts)
    f.create_dataset(f'RS_travel2clusters', data=travel2clusters)
    for robot in robots:
        print(f'{robot.id} --> {robot.tasks_init} & {robot.tasks_final} & {robot.tasks_finalized}')

        f.create_dataset(f'RS{robot.id}_Step_1', data=robot.tasks_init)
        f.create_dataset(f'RS{robot.id}_Step_2', data=robot.tasks_final)
        f.create_dataset(f'RS{robot.id}_Step_3', data=robot.tasks_finalized)

for victim in victims:
    print(victim.rescued)
    ax.scatter(victim.pos[0], victim.pos[1], c="blue", marker="s")
    ax.text(victim.pos[0], victim.pos[1], f'V{victim.id}')

# fig2, ax2 = plt.subplots(1, 1)
vor = Voronoi(clusters_coord)
voronoi_plot_2d(vor, ax)

plt.plot(dpi=1200)
plt.xticks([])
plt.yticks([])
ax.invert_yaxis()
plt.show()
