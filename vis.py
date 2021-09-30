import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def show3Dpose(channels, ax, radius=40, lcolor='red', rcolor='#0000ff'):
    vals = channels

    JOINTMAP = [
    [0,1],      #
    [1,2],      #
    [3,4],      #
    [4,5],      
    [6,0],
    [6,3],
    [6,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[6, 0], vals[6, 1], vals[6, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(10, -60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return [-RADIUS + xroot, RADIUS + xroot], [-RADIUS + yroot, RADIUS + yroot], [-RADIUS + zroot, RADIUS + zroot]

    # if lcolor=='blue' :
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    # else :
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("z")
    #     ax.set_zlabel("y")
