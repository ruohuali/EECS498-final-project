import numpy as np
import pybullet as p
import pybullet_data

from loc_utils import drawSphereMarker

def genNormalNoise(mean=(0, 0), cov=(1, 1)):
    """@return noise ~ (2, 1)"""
    mu = np.array(mean)
    sigma = np.diag(cov)
    noise = np.random.multivariate_normal(mu, sigma, 1).T
    # noise = np.concatenate(( noise, np.zeros((1, 1)) ), axis=0)
    return noise


def genMultNormalNoise(means=((0.2, 0.2), (-0.2, -0.2)), cov=(1, 1)):
    """@return noise ~ (2, 1)"""
    sigma = np.diag(cov)
    noise = np.zeros((2, 1))
    for mean in means:
        mu = np.array(mean)
        noise += np.random.multivariate_normal(mu, sigma, 1).T / len(means)
    return noise


def genUniformNoise(center=0, length=1):
    """@return noise ~ (2, 1)"""
    low = center - length / 2
    high = center + length / 2
    noise = np.random.uniform(low, high, size=(2, 1))
    return noise



