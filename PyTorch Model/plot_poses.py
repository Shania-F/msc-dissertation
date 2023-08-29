import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH = 'runs_TurtlebotTourCVSSP/run 3 - moar data'
pose_true = pd.read_csv(PATH + '/pose_true.csv')
pose_estim = pd.read_csv(PATH + '/pose_estim.csv')

# read ALL pos values
N = len(pose_true)
position_true = pose_true.iloc[:N, 0:3].values
position_estim = pose_estim.iloc[:N, 0:3].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(position_true[:N, 0], position_true[:N, 1], position_true[:N, 2], c='r', marker='o', label='truth')
ax.scatter(position_estim[:N, 0], position_estim[:N, 1], position_estim[:N, 2], c='b', marker='o', label='estimation')

# # connect the true and estim points with green lines
# for i in range(position_true.__len__()):
#    position_set = np.vstack((position_true[i,:], position_estim[i,:]))
#    ax.plot(position_set[:,0], position_set[:,1], position_set[:,2], color='green', linewidth=0.5)

ax.legend()

# plt.axis('scaled')
plt.show()
