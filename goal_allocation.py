import numpy as np
from scipy.optimize import linear_sum_assignment


def make_span_calc(robot_list, victim_list):
    for robot in robot_list:
        robot.travel_cost()
        for victim_id in robot.tasks_init:
            for cap in victim_list[victim_id].capabilities:
                if cap in robot.capabilities:
                    # Add the time for the accomplished task to the robot's make span
                    robot.make_span.append(victim_list[victim_id].make_span[cap])
                else:
                    # Accumulate the robot's abort time for the tasks he couldn't accomplish in the cluster
                    robot.abort_time += victim_list[victim_id].make_span[cap]


def time_cost(robot_list, victims_new, alpha=.2, beta=.25, gamma=.55):
    num_robots = len(robot_list)
    num_rem_tasks = len(victims_new)
    cost = np.ones((num_robots, num_rem_tasks))

    for robot in robot_list:
        for victim_id, victim in enumerate(victims_new):
            if victim.id in robot.tasks:  # Check with the capability analyser output
                cost[robot.id, victim_id] = ((gamma * np.max(robot.make_span)) +
                                             (alpha * robot.travel_time) +
                                             (beta * robot.abort_time / robot.num_sensors))
            else:
                cost[robot.id, victim_id] = 1e20

    return cost


def final_allocation(robot_list, victims_new):
    cost = time_cost(robot_list, victims_new, alpha=.2, beta=.25, gamma=.55)
    robots_opt, victims_opt = linear_sum_assignment(cost)
    for victim, robot in enumerate(robots_opt):
        robot_list[robot].tasks_final.append(victims_new[victims_opt[victim]].id)
        victims_new[victims_opt[victim]].rescued = True
