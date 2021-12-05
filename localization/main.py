import pybullet as p
import time
import pybullet_data
import numpy as np

from config import ROBOT_START_COORD, ROBOT_CMDS, TABLE1_COORD, TABLE2_COORD, SENSOR_NOISE_FUNC, MOTION_NOISE_FUNC
from loc_utils import drawSphereMarker
from loc_utils import pose2Coord, coord2Pose, cmd2PoseChange, updateCoordByCmd
from filters import kalmanFilter


def gps(body_id, noise_func):
    """@return noisy pose ~ (2, 1) is true coord + noise"""
    true_coord, _ = p.getBasePositionAndOrientation(body_id)  # tuple(x, y, z)
    true_pos = coord2Pose(true_coord)
    noise = noise_func()
    noisy_pose = true_pos + noise
    return noisy_pose


def motor(cmd, noise_func):
    """
    @note
    The original formulation is
    true cmd = received cmd + noise
    which implies the instruction is cmd which is perceived and add a noise on top which is what's truly been done
    this func uses
    received cmd = true cmd - noise
    which implies true cmd is known and received cmd is only used to trick filter
    """
    u = cmd2PoseChange(cmd)
    true_cmd = u
    noise = noise_func()
    received_cmd = true_cmd - noise
    return received_cmd


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
    mu = np.array([ROBOT_START_COORD[:2]]).T
    sigma = np.eye(2)
    print("shape", mu.shape, mu, sigma.shape, sigma)
    for step_idx, cmd in enumerate(ROBOT_CMDS):
        print("-" * 50)
        print("step", step_idx)

        # physics engine do cmd
        robot_cur_coord = updateCoordByCmd(robot_cur_coord, cmd)
        p.resetBasePositionAndOrientation(robot_id, robot_cur_coord, start_orientation)
        p.stepSimulation()
        true_coord, _ = p.getBasePositionAndOrientation(robot_id)
        drawSphereMarker([true_coord[0], true_coord[1], 1], 0.05, (0, 1, 0, .8))  # green is gt

        # filter try to estimate pose
        z = gps(robot_id, SENSOR_NOISE_FUNC)
        u = motor(cmd, MOTION_NOISE_FUNC)
        mu, sigma = kalmanFilter(mu, sigma, z, u)
        print("shape", mu.shape, sigma.shape, u.shape, z.shape)
        drawSphereMarker([mu[0, :], mu[1, :], 1], 0.05, (1, 0, 0, .8))  # red is gt

        # print the pos
        print("true pos", true_coord)
        print("robot cur pos", robot_cur_coord)
        print("noisy pos", z)
        print("estimation", mu)

        time.sleep(1. / 4.)

    while True:
        pass

    p.disconnect()


if __name__ == "__main__":
    main()
