import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points(predicted_points, measured_points):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the points to the plot
    for point in predicted_points:
        ax.scatter(point[0], point[1], point[2], c='b', marker='o')
    
    # Add the second set of points to the plot
    for point in measured_points:
        ax.scatter(point[0], point[1], point[2], c='r', marker='^')

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def accuracy(predicted_points, measured_points):
    # Calculate the Euclidean distance between each corresponding pair of points
    distances = [math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2) for p, m in zip(predicted_points, measured_points)]

    # Calculate the average distance (accuracy) between the two sets of points
    accuracy = sum(distances) / len(distances)

    return accuracy