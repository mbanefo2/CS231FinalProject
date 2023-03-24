import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points_3D(predicted_points, measured_points):
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

def plot_points_2D(predicted_points, measured_points):
    # Create a 2D plot
    fig, ax = plt.subplots()

    # Add the predicted points to the plot
    pred_xs = [point[0] for point in predicted_points]
    pred_ys = [point[1] for point in predicted_points]
    ax.scatter(pred_xs, pred_ys, c='b', marker='o', label='Predicted')

    # Add the measured points to the plot
    meas_xs = [point[0] for point in measured_points]
    meas_ys = [point[1] for point in measured_points]
    ax.scatter(meas_xs, meas_ys, c='r', marker='x', label='Measured')

    # Set the labels for the axes
    ax.set_xlabel('X axis pixels', fontsize=12)
    ax.set_ylabel('Y axis pixels', fontsize=12)

    # Set the title for the plot
    ax.set_title('Kalman Filter Predictions')

    # Add a legend to the plot
    ax.legend(fontsize=12)
    
    ax.grid(True)

    # Show the plot
    plt.show()

def plot_points_on_image(img, predicted_points, measured_points):
    # Create a 2D plot
    fig, ax = plt.subplots()
    
    ax.imshow(img)

    # Add the predicted points to the plot
    pred_xs = [point[0] for point in predicted_points]
    pred_ys = [point[1] for point in predicted_points]
    ax.scatter(pred_xs, pred_ys, c='b', marker='o', label='Predicted')

    # Add the measured points to the plot
    meas_xs = [point[0] for point in measured_points]
    meas_ys = [point[1] for point in measured_points]
    ax.scatter(meas_xs, meas_ys, c='r', marker='x', label='Measured')

    # Set the labels for the axes
    ax.set_xlabel('X axis pixels', fontsize=12)
    ax.set_ylabel('Y axis pixels', fontsize=12)

    # Set the title for the plot
    ax.set_title('Kalman Filter Predictions')

    # Add a legend to the plot
    ax.legend(fontsize=12)
    
    ax.grid(True)

    # Show the plot
    plt.show()

def accuracy(predicted_points, measured_points):
    # Calculate the Euclidean distance between each corresponding pair of points
    distances = [math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2) for p, m in zip(predicted_points, measured_points)]

    # Calculate the average distance (accuracy) between the two sets of points
    accuracy = sum(distances) / len(distances)

    return accuracy

def compute_accuracy(predicted_points, measured_points, threshold):
    num_points = len(predicted_points)
    num_accurate = 0
    
    for i in range(num_points):
        distance = euclidean_distance(predicted_points[i], measured_points[i])
        # print(distance)
        if distance < threshold:
            num_accurate += 1
            
    return (num_accurate / num_points) * 100

def euclidean_distance(predicted, measured):
    return np.linalg.norm(predicted - measured)

def compute_average_error(estimated_points, ground_truth_points):
    num_points = len(estimated_points)
    total_error = 0
    
    for i in range(num_points):
        distance = euclidean_distance(estimated_points[i], ground_truth_points[i])
        total_error += distance
        
    average_error = total_error / num_points
    return average_error
