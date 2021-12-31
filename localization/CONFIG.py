import matplotlib.pyplot as plt
import numpy as np
from noises import genNormalNoise, genMultNormalNoise, genUniformNoise

ROBOT_START_COORD = [0, 0, 0]
TABLE1_COORD = [6, 0, 0]
TABLE2_COORD = [12, 0, 0]
TABLE3_COORD = [6, 7, 0]
TABLE4_COORD = [12, 7, 0]
TABLE5_COORD = [6, -7, 0]
TABLE6_COORD = [12, -7, 0]
DOOR_COORD = [15, -4, 0]

ROBOT_CMDS = [
    np.array([0, 0]),  # x 0 y 0

    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),  # x 4 y 0

    np.array([0, 1]),
    np.array([0, 1]),
    np.array([0, 1]),
    np.array([0, 1]),  # x 4 y 4

    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),  # x 8 y 0
    np.array([1, 0]),  # x 9 y 0

    np.array([0, -1]),
    np.array([0, -1]),
    np.array([0, -1]),
    np.array([0, -1]),  # x 8 y 0

    np.array([0, -1]),
    np.array([0, -1]),
    np.array([0, -1]),
    np.array([0, -1]),  # x 8 y -4

    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),  # x 12 y -4
]


# experiment 1
# SENSOR_NOISE_FUNC = genNormalNoise
# SENSOR_NOISE_ARGS = {"mean": [0, 0], "cov": [1, 1]}
# MOTION_NOISE_FUNC = genNormalNoise
# MOTION_NOISE_ARGS = {"mean": [0, 0], "cov": [0.1, 0.1]}

# experiment 2
# SENSOR_NOISE_FUNC = genUniformNoise
# SENSOR_NOISE_ARGS = {"center": 0, "length": 3}
# MOTION_NOISE_FUNC = genUniformNoise
# MOTION_NOISE_ARGS = {"center": 0, "length": 1}

# experiment 3
# SENSOR_NOISE_FUNC = genUniformNoise
# SENSOR_NOISE_ARGS = {"center": 0, "length": 3}
# MOTION_NOISE_FUNC = genUniformNoise
# MOTION_NOISE_ARGS = {"center": 0, "length": 1}

# experiment 4
SENSOR_NOISE_FUNC = genUniformNoise
SENSOR_NOISE_ARGS = {"center": 3, "length": 6}
MOTION_NOISE_FUNC = genUniformNoise
MOTION_NOISE_ARGS = {"center": 0, "length": 1}

if __name__ == "__main__":
    # x = genNormalSensorNoise()
    # print(x, x.shape)
    n = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 10)
    print(n.shape)
