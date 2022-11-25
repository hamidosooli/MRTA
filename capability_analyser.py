import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import linear_sum_assignment

from robot import Robot
from victim import Victim


def capability_analyser(robot_list, victim_list):
    for robot in robot_list:
        for victim in victim_list:
            for capability in robot.capabilities:
                if capability in victim.capabilities:
                    robot.tasks.append(victim.id)
                    break


def victim_clustering(num_clusters, victim_list):
    """
    :param num_clusters: is equal to the number of regions/rooms in the environment
    :param victim_list: list of the victims in the environment
    :return:
    """
    victims_pos = []
    for victim in victim_list:
        victims_pos.append(victim.pos)

    # k-means
    clusters_coord = kmeans(victims_pos, num_clusters)[0]
    clusters = vq(victims_pos, clusters_coord)[0]
    # assign victim into the relevant cluster
    for cluster_idx, cluster in enumerate(clusters):
        victim_list[cluster_idx].cluster_id = cluster
        victim_list[cluster_idx].cluster_dist = np.subtract(victim_list[cluster_idx].pos, clusters_coord[cluster])

    return clusters, clusters_coord


def weight_calc(robot_list, clusters, num_clusters):
    cost = np.zeros((len(robot_list), num_clusters))
    for robot in robot_list:
        for victim_id, cluster_id in enumerate(clusters):
            if victim_id in robot.tasks:
                robot.w_rc[cluster_id] += 1
        cost[robot.id, :] = 1 / robot.w_rc
    return cost


def robot_distribution(robot_list, victim_list, clusters, clusters_coord):
    num_clusters = len(clusters_coord)
    cost = weight_calc(robot_list, clusters, num_clusters)
    robots_opt, clusters_opt = linear_sum_assignment(cost)
    victim_list_update = victim_list.copy()
    for robot_idx, robot in enumerate(robot_list):
        for cluster_idx, cluster in enumerate(clusters):
            if cluster == clusters_opt[robot_idx]:
                if victim_list[cluster_idx].id in robot.tasks:  # check for the ability to rescue (not sure about this!)
                    robot.tasks_init.append(victim_list[cluster_idx].id)
                    robot.tasks_init_dist.append(victim_list[cluster_idx].cluster_dist)
                    victim_list[cluster_idx].rescued = True
                    victim_list_update.remove(victim_list[cluster_idx])
                robot.pos = clusters_coord[cluster]
        Manhattan_to_Euclidean = np.linalg.norm(np.asarray(robot.tasks_init_dist), axis=1)
        # Sort the assignment on the basis of distance to the cluster coordination
        robot.tasks_init = [victim_id for _, victim_id in sorted(zip(Manhattan_to_Euclidean,
                                                                     robot.tasks_init))]

    return victim_list_update


if __name__ == '__main__':
    v1 = Victim(0, [2., 2.], ['DebrisRemover', 'OxygenCylinder'])
    v2 = Victim(1, [0., 13.], ['Diffuser', 'Manipulator'])
    v3 = Victim(2, [7., 15.], ['DebrisRemover', 'FireExtinguisher'])
    v4 = Victim(3, [7., 1.], ['OxygenCylinder', 'Manipulator'])
    v5 = Victim(4, [7., 10.], ['FirstAids', 'DebrisRemover'])
    v6 = Victim(5, [6., 12.], ['Manipulator', 'FireExtinguisher'])
    v7 = Victim(6, [5., 7.], ['FirstAids', 'Manipulator'])
    v8 = Victim(7, [9., 19.], ['FirstAids', 'Diffuser'])
    v9 = Victim(8, [15., 14.], ['DebrisRemover', 'Diffuser'])
    v10 = Victim(9, [19., 11.], ['FirstAids', 'Manipulator'])

    victims = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

    r1 = Robot(0, [0., 0.], 0.2, [], 4, [], ['Diffuser', 'DebrisRemover'])
    r2 = Robot(1, [0., 19.], 0.3, [], 4, [], ['FirstAids', 'OxygenCylinder', 'Manipulator'])
    r3 = Robot(2, [19., 0.], 0.4, [], 4, [], ['Manipulator'])
    r4 = Robot(3, [19., 19.], 0.3, [], 4, [], ['FirstAids', 'Diffuser'])

    robots = [r1, r2, r3, r4]

    capability_analyser(robots, victims)
    print(f'Robot 1 tasks based on capabilities: {r1.tasks}\n'
          f'Robot 2 tasks based on capabilities: {r2.tasks}\n'
          f'Robot 3 tasks based on capabilities: {r3.tasks}\n'
          f'Robot 4 tasks based on capabilities: {r4.tasks}\n')

    clusters, clusters_coord = victim_clustering(4, victims)
    victims_new = robot_distribution(robots, victims, clusters, clusters_coord)

    for idx, cluster in enumerate(clusters_coord):
        plt.scatter(cluster[0], cluster[1], c="red", marker="^")
        plt.text(cluster[0], cluster[1], f'cluster id: {idx}')
    print(clusters_coord, clusters)
    for robot in robots:
        print(robot.tasks_init)
    for victim in victims:
        print(victim.rescued)
        plt.scatter(victim.pos[0], victim.pos[1], c="blue", marker="s")
        plt.text(victim.pos[0], victim.pos[1], f'victim id: {victim.id}')
    plt.xlim([-1, 20])
    plt.ylim([-1, 20])
    plt.show()
