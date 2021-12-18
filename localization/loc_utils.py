import pybullet as p
import numpy as np


'''
@note
pose ~ (2, 1)
coord ~ (3, )
cmd ~ (2, )
'''
def pose2Coord(pose):
    coord = np.array([pose[0,:], pose[1,:], 0])
    return coord
def coord2Pose(coord):
    pose = np.array([coord[:2]]).T
    return pose
def cmd2PoseChange(cmd):
    pose = np.array([cmd]).T
    return pose
def updateCoordByCmd(cur_coord, cmd):
    coord = cur_coord + np.append(cmd, 0)
    return coord


def drawSphereMarker(position, radius, color):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id


def drawSphereMarker4Particles(particle_filter, k=100):
    particles = particle_filter.particles.copy()
    weights = particle_filter.weights.copy()
    topk_ids = np.argpartition(weights, -k)[-k:]
    particles = particles[topk_ids]
    for i, particle in enumerate(particles):
        drawSphereMarker([particle[0], particle[1], 1], 0.03, (0, 0.5, 0.7, 0.6))


def drawSphereMarker4Gaussian(mu, sigma, k=100):
    samples = np.random.multivariate_normal(mu.squeeze(), sigma, k)
    for i, sample in enumerate(samples):
        drawSphereMarker([sample[0], sample[1], 1], 0.03, (0.7, 0.5, 0, 0.6))


def drawNoise(noise_func, noise_args):
    drawSphereMarker([0, 0, 1], 0.1, (1, 0.3, 0.3, 1))
    for _ in range(1000):
        noise = noise_func(**noise_args)
        noise = list(noise.reshape(-1))
        noise.append(1)
        drawSphereMarker(noise, 0.05, (1, 0, 0.5, 0.3))
