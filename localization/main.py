import pdb

import pybullet as p
import time
import pybullet_data
import numpy as np
from pdb import set_trace

from CONFIG import ROBOT_START_COORD, ROBOT_CMDS, TABLE1_COORD, TABLE2_COORD, TABLE3_COORD, TABLE4_COORD, \
    TABLE5_COORD, TABLE6_COORD, DOOR_COORD, \
    SENSOR_NOISE_FUNC, SENSOR_NOISE_ARGS, MOTION_NOISE_FUNC, MOTION_NOISE_ARGS
from loc_utils import drawSphereMarker, drawSphereMarker4Particles, drawSphereMarker4Gaussian, drawNoise
from loc_utils import pose2Coord, coord2Pose, cmd2PoseChange, updateCoordByCmd
from filters import KalmanFilter, ParticleFilter


def gps(body_id, noise_func, noise_args):
    """@return noisy pose ~ (2, 1) is true coord + noise"""
    true_coord, _ = p.getBasePositionAndOrientation(body_id)  # tuple(x, y, z)
    true_pos = coord2Pose(true_coord)
    noise = noise_func(**noise_args)
    noisy_pose = true_pos + noise
    return noisy_pose


def motor(cmd, noise_func, noise_args):
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
    noise = noise_func(**noise_args)
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
    table1_id = p.loadURDF("table_collision/table.urdf", TABLE1_COORD, start_orientation)
    table2_id = p.loadURDF("table_collision/table.urdf", TABLE2_COORD, start_orientation)
    table3_id = p.loadURDF("table_collision/table.urdf", TABLE3_COORD, start_orientation)
    table4_id = p.loadURDF("table_collision/table.urdf", TABLE4_COORD, start_orientation)
    table5_id = p.loadURDF("table_collision/table.urdf", TABLE5_COORD, start_orientation)
    table6_id = p.loadURDF("table_collision/table.urdf", TABLE6_COORD, start_orientation)
    door_id = p.loadURDF("door.urdf", DOOR_COORD, p.getQuaternionFromEuler([0, 0, 1.5]))

    robot_cur_coord = ROBOT_START_COORD

    # init filters
    mu = coord2Pose(robot_cur_coord)
    sigma = np.eye(2)
    kf = KalmanFilter(mu, sigma)
    init_pos = coord2Pose(robot_cur_coord)
    pf = ParticleFilter(init_pos, particle_num=100)

    # init errors
    kf_l1_errors = []
    kf_l2_errors = []
    kf_final_error = 0.
    pf_l1_errors = []
    pf_l2_errors = []
    pf_final_error = 0.

    for step_idx, cmd in enumerate(ROBOT_CMDS):
        print("-" * 50)
        print("step", step_idx)

        # physics engine do cmd
        robot_cur_coord = updateCoordByCmd(robot_cur_coord, cmd)
        p.resetBasePositionAndOrientation(robot_id, robot_cur_coord, start_orientation)
        p.stepSimulation()
        true_coord, _ = p.getBasePositionAndOrientation(robot_id)
        drawSphereMarker([true_coord[0], true_coord[1], 2], 0.1, (0, 1, 0, .8))  # green is gt

        # filter try to estimate pose
        z = gps(robot_id, SENSOR_NOISE_FUNC, SENSOR_NOISE_ARGS)
        u = motor(cmd, MOTION_NOISE_FUNC, MOTION_NOISE_ARGS)
        drawSphereMarker([z[0, :], z[1, :], 2], 0.1, (0.5, 0.1, 0.5, .8))  # yellow is noisy sensor

        mu, sigma = kf(z, u)
        drawSphereMarker([mu[0, :], mu[1, :], 2], 0.1, (1, 0, 0, .8))  # red is kf
        # drawSphereMarker4Gaussian(mu, sigma)

        es = pf(z, u)
        drawSphereMarker([es[0, :], es[1, :], 2], 0.1, (0, 0, 1, .8))  # blue is pf
        # drawSphereMarker4Particles(pf)

        # update errors
        mu = mu.reshape(-1)
        es = es.reshape(-1)
        kf_l1_errors.append((abs(true_coord[0] - mu[0]) + abs(true_coord[1] - mu[1])))
        kf_l2_errors.append(((true_coord[0] - mu[0]) ** 2 + (true_coord[1] - mu[1]) ** 2) ** (0.5))
        kf_final_error = ((true_coord[0] - mu[0]) ** 2 + (true_coord[1] - mu[1]) ** 2) ** (0.5)
        pf_l1_errors.append((abs(true_coord[0] - es[0]) + abs(true_coord[1] - es[1])))
        pf_l2_errors.append(((true_coord[0] - es[0]) ** 2 + (true_coord[1] - es[1]) ** 2) ** (0.5))
        pf_final_error = ((true_coord[0] - es[0]) ** 2 + (true_coord[1] - es[1]) ** 2) ** (0.5)

        # pdb.set_trace()

    # show errors
    print("=" * 30)
    print("kf errors")
    print("MAE:", sum(kf_l1_errors) / len(kf_l1_errors))
    print("MSE:", sum(kf_l2_errors) / len(kf_l2_errors))
    print("FE:", kf_final_error)

    print("=" * 30)
    print("pf errors")
    print("MAE:", sum(pf_l1_errors) / len(pf_l1_errors))
    print("MSE:", sum(pf_l2_errors) / len(pf_l2_errors))
    print("FE:", pf_final_error)

    countdown = 30
    for _ in range(countdown):
        time.sleep(1)

    p.disconnect()


if __name__ == "__main__":
    main()
