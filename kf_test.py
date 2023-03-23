from kalman_filter import KalmanFilter
from util import plot_points, accuracy

    
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

plot_points(kf_points, measurements)
accur = accuracy(kf_points, measurements)
print(accur)
