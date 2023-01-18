import numpy as np
from scipy.optimize import linear_sum_assignment


def make_span_calc(robot_list, task_list):
    for robot in robot_list:
        robot.travel_cost()
        for task_id in robot.tasks_init:
            for cap in task_list[task_id].capabilities:
                if cap in robot.capabilities:
                    # Add the time for the accomplished task to the robot's make span
                    robot.make_span.append(task_list[task_id].make_span[cap])
                else:
                    # Accumulate the robot's abort time for the tasks he couldn't accomplish in the cluster
                    robot.abort_time += task_list[task_id].make_span[cap]


def time_cost(robot_list, tasks_new, alpha=.2, beta=.25, gamma=.55):
    num_robots = len(robot_list)
    num_rem_tasks = len(tasks_new)
    cost = np.ones((num_robots, num_rem_tasks))
    for robot in robot_list:
        if len(robot.make_span) == 0:
            robot.make_span.append(0)
        for task_id, task in enumerate(tasks_new):
            if task.id in robot.tasks:  # Check with the capability analyser output
                cost[robot.id, task_id] = ((gamma * np.max(robot.make_span)) +
                                           (alpha * robot.travel_time) +
                                           (beta * robot.abort_time / robot.num_sensors))
            else:
                cost[robot.id, task_id] = 1e20

    return cost


def final_allocation(robot_list, task_list, tasks_new):

    make_span_calc(robot_list, task_list)
    cost = time_cost(robot_list, tasks_new, alpha=.0, beta=.99, gamma=.01)
    robots_opt, tasks_opt = linear_sum_assignment(cost)
    for task, robot in enumerate(robots_opt):
        robot_list[robot].tasks_final.append(tasks_new[tasks_opt[task]].id)

        caps = [cap in robot_list[robot].capabilities for cap in tasks_new[tasks_opt[task]].capabilities]

        for cap_id, cap in enumerate(caps):
            if cap:
                tasks_new[tasks_opt[task]].rescued[cap_id] = True
                task_list[task_list.index(tasks_new[tasks_opt[task]])].rescued[cap_id] = True
    for task in tasks_new:
        if all(task.rescued):
            tasks_new.remove(task)

    return tasks_new


def finalizer(robot_list, task_list_new, task_list):
    for task in task_list:
        for idx, status in enumerate(task.rescued):
            if not status:
                dist = []
                for candid in task.candidates[idx]:
                    # Where is this (candidate) robot now
                    if len(robot_list[candid].tasks_final) > 0:
                        dist.append(task_list[robot_list[candid].tasks_final[-1]].pos)
                    elif len(robot_list[candid].tasks_init) > 0:
                        dist.append(task_list[robot_list[candid].tasks_init[-1]].pos)
                temp = np.inf
                id = np.nan
                for candid_id, d in enumerate(dist):
                    euc_dist = np.linalg.norm(np.subtract(d, task.pos))
                    if euc_dist < temp:
                        temp = euc_dist.copy()
                        id = task.candidates[idx][candid_id]
                robot_list[id].tasks_finalized.append(task.id)
                task.rescued[idx] = True
    for task in task_list_new:
        if all(task.rescued):
            task_list_new.remove(task)
    if len(task_list_new) and all(task_list_new[0].rescued):
        del task_list_new[0]
    return task_list_new
