import numpy as np
from kalman_filter import KalmanFilter
from util import plot_points_2D, compute_accuracy, compute_average_error

# Create an empty list to store arrays
# values_list = [
#     [314, 374],
#     [313, 376],
#     [313, 373],
#     [313, 371],
#     [313, 374],
#     [315, 376],
#     [314, 376],
#     [315, 378],
#     [315, 380],
#     [315, 383],
#     [316, 382],
#     [314, 382],
#     [316, 383],
#     [317, 383],
#     [317, 380],
#     [316, 380],
#     [314, 380],
#     [314, 378],
#     [315, 377],
#     [313, 377],
#     [313, 374],
#     [312, 373],
#     [312, 371],
#     [311, 373],
#     [310, 371],
#     [311, 370],
#     [309, 367],
#     [310, 368],
#     [309, 368],
#     [308, 367],
#     [309, 364],
#     [308, 362],
#     [308, 360],
#     [309, 360],
#     [307, 358],
#     [308, 357],
#     [308, 355],
#     [308, 352],
#     [307, 352],
#     [307, 351],
#     [306, 350],
#     [306, 348],
#     [306, 346],
#     [306, 345],
#     [306, 344],
#     [307, 344],
#     [306, 341],
#     [306, 339],
#     [307, 339],
#     [306, 328],
#     [306, 335],
#     [305, 320],
#     [301, 296],
#     [299, 275],
#     [298, 259],
#     [297, 248],
#     [296, 239],
#     [293, 229],
#     [292, 221],
#     [293, 217],
#     [291, 213],
#     [292, 214],
#     [290, 213],
#     [290, 211],
#     [289, 211],
#     [288, 213],
#     [288, 216],
#     [288, 221],
#     [290, 223],
#     [299, 224],
#     [306, 211],
#     [308, 209],
#     [315, 207],
#     [320, 206],
#     [324, 207],
#     [329, 206],
#     [336, 209],
#     [341, 211],
#     [347, 213],
#     [351, 221],
#     [356, 226],
#     [361, 232],
#     [365, 241],
#     [369, 251],
#     [373, 257],
#     [377, 268],
#     [381, 278],
#     [384, 293],
#     [388, 303],
#     [391, 314],
#     [384, 309],
#     [370, 290],
#     [358, 275],
#     [350, 259],
#     [341, 251],
#     [332, 245],
#     [322, 239],
#     [316, 236],
#     [308, 234],
#     [301, 232],
#     [295, 229],
#     [292, 229],
#     [291, 229],
#     [283, 230],
#     [281, 232],
#     [276, 234],
#     [270, 239]
# ]

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

# get the unique rows of the 2D array
unique_rows = np.unique(array2d, axis=0)

# convert the unique rows back to a list of arrays
# frame captures might be too quick and not much movement in shuttlecock
unique_arrays = [np.array(row) for row in unique_rows]
test_array = unique_arrays

kf = KalmanFilter(60)
kf_points = []

for point in test_array:
    kf.predict()
    mu, sigma = kf.update(point)
    kf_points.append(mu)

test_kf_points = []

for i in range(len(kf_points)):
    test_kf_points.append([kf_points[i][:2][0][0], kf_points[i][:2][1][0]])

plot_points_2D(test_kf_points, test_array)

new_accur = compute_accuracy(test_kf_points, test_array, 10)
print(f' New Accuracy {new_accur}')

avg_error = compute_average_error(test_kf_points, test_array)
print(f'This is the avg_error {avg_error}')
