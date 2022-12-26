def capability_analyser(robot_list, task_list):
    # Go through the list of robots
    for robot in robot_list:
        # Go through the list of tasks
        for task in task_list:
            # Find the robots that can satisfy all the requirements of a special task
            if all(cap in robot.capabilities for cap in task.capabilities):
                robot.tasks_full.append(task.id)

            # Assign tasks to the robots based on their requirements
            for capability in robot.capabilities:
                if capability in task.capabilities:
                    robot.tasks.append(task.id)
                    break

            # Find the robot candidates that can satisfy each requirement
            for capability in robot.capabilities:
                if capability in task.capabilities:
                    task.candidates[task.capabilities.index(capability)].append(robot.id)


def assistance(task_list):
    human_assistance = []
    for task in task_list:
        print(task.candidates)
        for candid_id, candid in enumerate(task.candidates):
            if len(candid) == 0:
                task.rescued[candid_id] = True
                human_assistance.append({task.id: task.capabilities[candid_id]})

    print(human_assistance)
    return human_assistance
