import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import linear_sum_assignment


def task_clustering(num_clusters, task_list):
    """
    :param num_clusters: is equal to the number of regions/rooms in the environment
    :param task_list: list of the tasks in the environment
    :return:
    """
    tasks_pos = []
    for task in task_list:
        tasks_pos.append(task.pos)

    # k-means
    clusters_coord = kmeans(tasks_pos, num_clusters)[0]
    clusters = vq(tasks_pos, clusters_coord)[0]
    # assign task into the relevant cluster
    for cluster_idx, cluster in enumerate(clusters):
        task_list[cluster_idx].cluster_id = cluster
        task_list[cluster_idx].cluster_dist = np.subtract(task_list[cluster_idx].pos, clusters_coord[cluster])

    return clusters, clusters_coord


def weight_calc(robot_list, task_list, clusters, beta=100):
    '''
    :param robot_list: list of the robot objects (rescue team)
    :param task_list:
    :param clusters: list of  the clusters calculated by task_clustering
    :param beta: Coefficient for fulfilled tasks
    :return: the cost that is the opposite of the number of tasks each robot can save at each cluster
    '''
    num_clusters = np.max(clusters) + 1
    cost = np.zeros((len(robot_list), num_clusters))
    for robot in robot_list:
        for task_id, cluster_id in enumerate(clusters):

            if task_id in robot.tasks:
                caps = [cap in robot.capabilities for cap in task_list[task_id].capabilities]
                robot.w_rc[cluster_id] += sum(caps)
            if task_id in robot.tasks_full:
                robot.w_rc[cluster_id] += beta * len(task_list[task_id].capabilities)

        cost[robot.id, :] = -robot.w_rc
    return cost


def robot_distribution(robot_list, task_list, clusters, clusters_coord):
    cost = weight_calc(robot_list, task_list, clusters)
    robots_opt, clusters_opt = linear_sum_assignment(cost)
    task_list_update = task_list.copy()
    for robot_idx, robot in enumerate(robot_list):
        for cluster_idx, cluster in enumerate(clusters):
            if cluster == clusters_opt[robot_idx]:
                if task_list[cluster_idx].id in robot.tasks:  # check for the ability to rescue
                    # Assign the tasks to the robot for rescue
                    robot.tasks_init.append(task_list[cluster_idx].id)
                    robot.tasks_init_dist.append(task_list[cluster_idx].cluster_dist)
                    caps = [cap in robot.capabilities for cap in task_list[cluster_idx].capabilities]

                    for cap_id, cap in enumerate(caps):
                        if cap:
                            task_list[cluster_idx].rescued[cap_id] = True

                    # if all(task_list[cluster_idx].rescued):
                        # Update the list of the remaining tasks and set the task's rescue flag True
                    task_list_update.remove(task_list[cluster_idx])

                # Send the robot to cluster's center
                robot.pos = clusters_coord[cluster]
        Manhattan_to_Euclidean = np.linalg.norm(np.asarray(robot.tasks_init_dist), axis=1)
        # Sort the assignment on the basis of distance to the cluster coordination
        robot.tasks_init = [task_id for _, task_id in sorted(zip(Manhattan_to_Euclidean,
                                                                     robot.tasks_init))]

    return task_list_update
