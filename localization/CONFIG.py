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



def genNormalSensorNoise(mean=[0, 0], cov=[1, 1]):
    """@return noise ~ (2, 1)"""
    mu = np.array(mean)
    sigma = np.diag(cov)
    noise = np.random.multivariate_normal(mu, sigma, 1).T
    # noise = np.concatenate(( noise, np.zeros((1, 1)) ), axis=0)
    return noise

def genNormalMotionNoise(mean=[0, 0], cov=[0.1, 0.1]):
    """@return noise ~ (2, 1)"""
    mu = np.array(mean)
    sigma = np.diag(cov)
    noise = np.random.multivariate_normal(mu, sigma, 1).T
    # noise = np.concatenate(( noise, np.zeros((1, 1)) ), axis=0)
    return noise

def gen4MNormalSensorNoise():
    noise = np.zeros((2, 1))
    mean = [1, 1]
    cov = [1, 1]
    noise += genNormalSensorNoise(mean, cov)
    mean = [-1, 1]
    cov = [1, 1]
    noise += genNormalSensorNoise(mean, cov)
    mean = [1, -1]
    cov = [1, 1]
    noise += genNormalSensorNoise(mean, cov)
    mean = [-1, -1]
    cov = [1, 1]
    noise += genNormalSensorNoise(mean, cov)

    mean = [2, -2]
    cov = [0.5, 0.5]
    noise += genNormalSensorNoise(mean, cov)
    mean = [-2, 2]
    cov = [0.5, 0.5]
    noise += genNormalSensorNoise(mean, cov)
    mean = [-2, -2]
    cov = [0.5, 0.5]
    noise += genNormalSensorNoise(mean, cov)
    mean = [2, 2]
    cov = [0.5, 0.5]
    noise += genNormalSensorNoise(mean, cov)
    return noise


# SENSOR_NOISE_FUNC = genNormalSensorNoise
SENSOR_NOISE_FUNC = gen4MNormalSensorNoise
MOTION_NOISE_FUNC = genNormalSensorNoise

if __name__ == "__main__":
    # x = genNormalSensorNoise()
    # print(x, x.shape)
    noise = np.random.multivariate_normal([1, 1], [[1,0],[0,1]], 10)
    print(noise.shape)
