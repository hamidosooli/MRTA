def capability_analyser(robot_list, victim_list):
    for robot in robot_list:
        for victim in victim_list:
            for capability in robot.capabilities:
                if capability in victim.capabilities:
                    robot.tasks.append(victim.id)
                    break
