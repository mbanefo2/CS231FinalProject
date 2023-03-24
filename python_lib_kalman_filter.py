from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from util import plot_points_2D, accuracy, compute_accuracy, compute_average_error

# Define the initial state of the filter
x0 = np.array([[0], [0], [0], [0]])
P0 = np.eye(4) * 1000  # initial state covariance
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # state transition matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])  # observation matrix
R = np.eye(2) * 10  # measurement noise covariance
Q = Q_discrete_white_noise(dim=4, dt=1.0, var=0.1)  # process noise covariance

# Create the Kalman filter object
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = x0
kf.P = P0
kf.F = F
kf.H = H
kf.R = R
kf.Q = Q

# Update the filter with a new measurement
def update(x, y):
    z = np.array([[x], [y]])
    kf.predict()
    kf.update(z)
    return kf.x[0][0], kf.x[1][0]

values_list = [
    [391, 314],
    [384, 309],
    [370, 290],
    [358, 275],
    [350, 259],
    [341, 251],
    [332, 245],
    [322, 239],
    [316, 236],
    [308, 234],
    [301, 232],
    [295, 229],
    [292, 229],
    [291, 229],
    [283, 230],
    [281, 232],
    [276, 234],
    [270, 239]
]

array2d = np.vstack(values_list)
# print(array2d.shape)

# get the unique rows of the 2D array
unique_rows = np.unique(array2d, axis=0)

# convert the unique rows back to a list of arrays
unique_arrays = [np.array(row) for row in unique_rows]

kf_points = []
for point in unique_arrays:
    x,y = update(point[0], point[1])
    new_point = [x,y]
    # print(x)
    # print(y)
    print(new_point)
    
    
    kf_points.append(new_point)

test_kf_points = kf_points #[:-1]
print(type(test_kf_points))

accur = accuracy(test_kf_points, unique_arrays)
print(f'Accuracy {accur}')

new_accur = compute_accuracy(test_kf_points, unique_arrays, 10)
print(f' New Accuracy {new_accur}')

avg_error = compute_average_error(test_kf_points, unique_arrays)
print(f'This is the avg_error {avg_error}')