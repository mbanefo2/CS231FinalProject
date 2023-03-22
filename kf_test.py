from kalman_filter import KalmanFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points(points, points2):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the points to the plot
    for point in points:
        ax.scatter(point[0], point[1], point[2], c='b', marker='o')
    
    # Add the second set of points to the plot
    for point in points2:
        ax.scatter(point[0], point[1], point[2], c='r', marker='^')

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()
    
kf = KalmanFilter(60)

measurements = [
    [0.5, 0.5, 0.1],
    [1.5, 1.5, 1.5],
    [2.7, 3.0, 2.5],
    [3.5, 4.5, 3.5],
    [4.2, 5.5, 4.8],
    [5.0, 7.0, 6.0],
    [5.8, 8.5, 7.2],
    [6.0, 10.0, 8.0],
    [5.5, 11.2, 7.5],
    [4.7, 12.0, 6.8],
    [3.5, 12.5, 6.0],
    [2.5, 11.5, 5.0],
    [1.5, 10.0, 4.2],
    [0.8, 8.5, 3.5],
    [0.3, 7.0, 3.0],
    [0.7, 5.5, 2.5],
    [1.5, 4.0, 2.0],
    [2.5, 2.5, 1.5],
    [3.0, 1.5, 1.0],
    [3.5, 0.5, 0.1],
    [3.5, 0, 0.1],
    ]
count = 0
kf_points = [[0.5, 0.5, 0.1]]
for point in measurements:
    kf.update(point)
    hit_floor, mu, sigma = kf.predict()
    
    if hit_floor:
        count = count + 1
        print(f'Ball hit the floor at point {point}')
    
    kf_points.append(mu)

print(f'Ball hit floor {count} times')
print(f'Predicted points {kf_points}')

plot_points(measurements, kf_points)
