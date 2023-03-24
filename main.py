import numpy as np
import cv2
from kalman_filter import KalmanFilter
from court_detector import CourtBoundDetector
from image_processor import ImageProcessor
from util import plot_points_2D, compute_accuracy, compute_average_error, plot_points_on_image

# Shuttlecock positions from YOLO on ROBOFLOW image frames 2_906-2_938
yolo_output = [
    [647,322],
    [654,308],
    [664,295],
    [710,281],
    [719,283],
    [710,259],
    [607,310],
    [608,313],
    [614,313],
    [584,330],
    [566,341],
    [552,354],
    [548,355],
    [519,386],
    [503,406],
    [503,403],
    [492,421],
    [477,439],
    [479,441],
    [465,457],
    [454,475],
    [448,421],
    [446,379],
    [450,382],
    [449,326],
    [451,328],
    [448,305],
    [449,288],
    [448,276],
    [452,266],
    [450,249],
    [453,244],
    [455,241]
]

# Ensure measurements are unique
array2d = np.vstack(yolo_output)
unique_rows = np.unique(array2d, axis=0)

# convert the unique rows back to a list of arrays
# frame captures might be too quick and not much movement in shuttlecock
# Hence the need to ensure each value is indeed a unique measurement as in reality
# The shuttlecock does not hover in place
unique_yolo_measurements = [np.array(row) for row in unique_rows]

################################ Kalman filter projections ##################################
kf = KalmanFilter(60)

# Initialize output lists
kf_points = []
result_kf_points = []

# Perform updates and predictions
for point in unique_yolo_measurements:
    kf.predict()
    mu, sigma = kf.update(point)
    kf_points.append(mu)

# Parse the resulting output
for i in range(len(kf_points)):
    result_kf_points.append([kf_points[i][:2][0][0], kf_points[i][:2][1][0]])

# Plot predicted vs measured points
plot_points_2D(result_kf_points, unique_yolo_measurements)

accuracy = compute_accuracy(result_kf_points, unique_yolo_measurements, 10)
print(f'Accuracy of Kalman Filter {accuracy}')

avg_error = compute_average_error(result_kf_points, unique_yolo_measurements)
print(f'Average error for Kalman Filter {avg_error}')

################################ Court Boundary Extraction ##################################
filename = '/Users/bryanmbanefo/Desktop/test_grey_scale/Test_small_shuttle_cock.v1i.tensorflow/train/youtube-131_jpg.rf.e07beeae41664b9209347ff9fb94eaed.jpg'

x = result_kf_points[0][0]
y = result_kf_points[0][1]

img = cv2.imread(filename)
cbd = CourtBoundDetector(img)
img = cbd.resize_img(img)
img = cbd.bilateral_filter(img, 9)
lum = cbd.luminance(img)
white = cbd.whiteline_detection(lum, 150, 25, 3)
hough_lines = cbd.hough(white)
hough = cbd.draw_lines(img, hough_lines, (int(x), int(y)), True)

for i in [lum, white, hough]:
    cv2.imshow('image', i)   
    key = cv2.waitKey(0)
    if key == 27: 
        cv2.destroyAllWindows()

################################ Draw Measured and Predicted path on image ##################################
plot_points_on_image(cbd.og_img, result_kf_points, unique_yolo_measurements)
