# Uncomment for 6-plot 2D image
# import matplotlib
# matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

PATH = 'runs_TurtlebotTourCVSSP/run 3 - moar data'
pose_true = pd.read_csv(PATH + '/pose_true.csv')
pose_estim = pd.read_csv(PATH + '/pose_estim.csv')

# fig = plt.figure(figsize=(15, 10))
N = 25  # start from which test sample

# read ori values
for i in range(6):
    true_quaternion = pose_true.iloc[N, 3:7].values
    predicted_quaternion = pose_estim.iloc[N, 3:7].values
    N = N+1

    # Convert quaternions to rotation matrices (when multiplied with a vector produced the rotated vector
    true_rotation = Rotation.from_quat(true_quaternion).as_matrix()
    print(true_rotation)
    predicted_rotation = Rotation.from_quat(predicted_quaternion).as_matrix()

    # Extract x, y, z axes from rotation matrices
    true_x_axis = true_rotation[:, 0]
    true_y_axis = true_rotation[:, 1]
    true_z_axis = true_rotation[:, 2]  # forward-facing or nose direction

    predicted_x_axis = predicted_rotation[:, 0]
    predicted_y_axis = predicted_rotation[:, 1]
    predicted_z_axis = predicted_rotation[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(2, 3, i+1, projection='3d')

    # Plot the true orientation (red axes)
    ax.quiver(0, 0, 0, true_x_axis[0], true_x_axis[1], true_x_axis[2], color='r', linestyle='dashed')
    ax.quiver(0, 0, 0, true_y_axis[0], true_y_axis[1], true_y_axis[2], color='r', linestyle='dotted')
    ax.quiver(0, 0, 0, true_z_axis[0], true_z_axis[1], true_z_axis[2], color='r')

    # Plot the predicted orientation (green axes)
    ax.quiver(0, 0, 0, predicted_x_axis[0], predicted_x_axis[1], predicted_x_axis[2], color='g', linestyle='dashed')
    ax.quiver(0, 0, 0, predicted_y_axis[0], predicted_y_axis[1], predicted_y_axis[2], color='g', linestyle='dotted')
    ax.quiver(0, 0, 0, predicted_z_axis[0], predicted_z_axis[1], predicted_z_axis[2], color='g')

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('True vs Predicted Quaternion Orientations')

    # ax.set_title(f'Subplot {i + 1}')
    # plt.tight_layout()  # prevent overlapping titles and labels

    # Show the plot
    plt.show()
    # plt.savefig(PATH + '/ori_subplots.png')
