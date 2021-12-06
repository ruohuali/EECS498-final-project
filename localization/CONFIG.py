import matplotlib.pyplot as plt
import numpy as np
from noises import genNormalNoise, genMultNormalNoise, genUniformNoise

ROBOT_START_COORD = [0, 0, 0]
TABLE1_COORD = [6, 0, 0]
TABLE2_COORD = [10, 0, 0]

ROBOT_CMDS = [np.array([0, 0]),   # x 0 y 0

              np.array([1, 0]),
              np.array([1, 0]),
              np.array([1, 0]),
              np.array([1, 0]),   # x 4 y 0

              np.array([0, 1]),
              np.array([0, 1]),
              np.array([0, 1]),
              np.array([0, 1]),   # x 4 y 4

              np.array([1, 0]),
              np.array([1, 0]),
              np.array([1, 0]),
              np.array([1, 0]),  # x 8 y 0

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


# SENSOR_NOISE_FUNC = genNormalNoise
# SENSOR_NOISE_ARGS = {"mean": [0, 0], "cov": [1, 1]}
SENSOR_NOISE_FUNC = genUniformNoise
SENSOR_NOISE_ARGS = {"center": -2, "length": 2}
# MOTION_NOISE_FUNC = genNormalNoise
# MOTION_NOISE_ARGS = {"mean": [0, 0], "cov": [0.1, 0.1]}
MOTION_NOISE_FUNC = genUniformNoise
MOTION_NOISE_ARGS = {"center": 0, "length": 1}


if __name__ == "__main__":
    # x = genNormalSensorNoise()
    # print(x, x.shape)
    n = np.random.multivariate_normal([1, 1], [[1,0],[0,1]], 10)
    print(n.shape)
