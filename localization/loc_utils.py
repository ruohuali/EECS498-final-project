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
