import pybullet as p
import time
import pybullet_data
import numpy as np

from config import ROBOT_START_COORD, ROBOT_CMDS, TABLE1_COORD, TABLE2_COORD, NOISE_FUNC
from loc_utils import drawSphereMarker


def gps(body_id, noise_func):
    """@return noisy pose ~ (2, 1) is true coord + noise"""
    true_pos, _ = p.getBasePositionAndOrientation(body_id)   # tuple(x, y, z)
    true_pos = true_pos[:2]
    true_pos = np.array(true_pos).T
    noise = noise_func()
    noisy_pose = true_pos + noise
    return noisy_pose


def main():
    # set up the physics client
    physics_client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    # load the models
    plane_id = p.loadURDF("plane.urdf")
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # doesn't count so not matter
    robot_id = p.loadURDF("pr2_description/pr2.urdf", ROBOT_START_COORD, start_orientation)
    table1_id = p.loadURDF("table/table.urdf", TABLE1_COORD, start_orientation)
    table2_id = p.loadURDF("table/table.urdf", TABLE2_COORD, start_orientation)

    # drawSphereMarker([0, 0, 1], 0.1, (0, 0, 1, .5))
    # drawSphereMarker([0, 1, 0], 0.1, (0, 1, 0, .8))

    robot_cur_coord = ROBOT_START_COORD
    for step_idx, cmd in enumerate(ROBOT_CMDS):
        print("-" * 50)
        print("step", step_idx)

        # physics engine do cmd
        robot_cur_coord += np.append(cmd, 0)
        p.resetBasePositionAndOrientation(robot_id, robot_cur_coord, start_orientation)
        p.stepSimulation()

        # check the pos
        true_coord, _ = p.getBasePositionAndOrientation(robot_id)
        noisy_coord = gps(robot_id, NOISE_FUNC)
        print("true pos", true_coord)
        print("robot cur pos", robot_cur_coord)
        print("noisy pos", noisy_coord)

        time.sleep(1. / 4.)

    p.disconnect()


if __name__ == "__main__":
    main()
