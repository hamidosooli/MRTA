import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import linear_sum_assignment


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


def weight_calc(robot_list, victim_list, clusters, beta=100):
    '''
    :param robot_list: list of the robot objects (rescue team)
    :param victim_list:
    :param clusters: list of  the clusters calculated by victim_clustering
    :param beta: Coefficient for fulfilled victims
    :return: the cost that is the opposite of the number of victims each robot can save at each cluster
    '''
    num_clusters = np.max(clusters) + 1
    cost = np.zeros((len(robot_list), num_clusters))
    for robot in robot_list:
        for victim_id, cluster_id in enumerate(clusters):

            if victim_id in robot.tasks:
                caps = [cap in robot.capabilities for cap in victim_list[victim_id].capabilities]
                robot.w_rc[cluster_id] += sum(caps)
            if victim_id in robot.tasks_full:
                robot.w_rc[cluster_id] += beta * len(victim_list[victim_id].capabilities)

        cost[robot.id, :] = -robot.w_rc
    return cost


def robot_distribution(robot_list, victim_list, clusters, clusters_coord):
    cost = weight_calc(robot_list, victim_list, clusters)
    robots_opt, clusters_opt = linear_sum_assignment(cost)
    victim_list_update = victim_list.copy()
    for robot_idx, robot in enumerate(robot_list):
        for cluster_idx, cluster in enumerate(clusters):
            if cluster == clusters_opt[robot_idx]:
                if victim_list[cluster_idx].id in robot.tasks:  # check for the ability to rescue
                    # Assign the victims to the robot for rescue
                    robot.tasks_init.append(victim_list[cluster_idx].id)
                    robot.tasks_init_dist.append(victim_list[cluster_idx].cluster_dist)
                    caps = [cap in robot.capabilities for cap in victim_list[cluster_idx].capabilities]

                    for cap_id, cap in enumerate(caps):
                        if cap:
                            victim_list[cluster_idx].rescued[cap_id] = True

                    # if all(victim_list[cluster_idx].rescued):
                        # Update the list of the remaining victims and set the victim's rescue flag True
                    victim_list_update.remove(victim_list[cluster_idx])

                # Send the robot to cluster's center
                robot.pos = clusters_coord[cluster]
        Manhattan_to_Euclidean = np.linalg.norm(np.asarray(robot.tasks_init_dist), axis=1)
        # Sort the assignment on the basis of distance to the cluster coordination
        robot.tasks_init = [victim_id for _, victim_id in sorted(zip(Manhattan_to_Euclidean,
                                                                     robot.tasks_init))]

    return victim_list_update
