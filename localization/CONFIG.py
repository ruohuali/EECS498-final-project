import matplotlib.pyplot as plt
import numpy as np

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



def genNormalSensorNoise():
    """@return noise ~ (2, 1)"""
    MEAN = [0, 0]
    COV = [1, 1]
    mu = np.array(MEAN)
    sigma = np.diag(COV)
    noise = np.random.multivariate_normal(mu, sigma, 1).T
    # noise = np.concatenate(( noise, np.zeros((1, 1)) ), axis=0)
    return noise

def genNormalMotionNoise():
    """@return noise ~ (2, 1)"""
    MEAN = [0, 0]
    COV = [0.1, 0.1]
    mu = np.array(MEAN)
    sigma = np.diag(COV)
    noise = np.random.multivariate_normal(mu, sigma, 1).T
    # noise = np.concatenate(( noise, np.zeros((1, 1)) ), axis=0)
    return noise


SENSOR_NOISE_FUNC = genNormalSensorNoise
MOTION_NOISE_FUNC = genNormalMotionNoise

if __name__ == "__main__":
    # x = genNormalSensorNoise()
    # print(x, x.shape)
    noise = np.random.multivariate_normal([1, 1], [[1,0],[0,1]], 10)
    print(noise.shape)
