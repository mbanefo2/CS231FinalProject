import numpy as np
from camera import Camera



R = np.vstack([
                [0.9229406007, -0.05524817202, -0.3809570672],
                [-0.1025391938, 0.918605371, -0.3816410435],
                [0.3710341781, 0.3912950444, 0.8421530899]
                ])
T = np.array([-100.4858498, -44.85467922, 289.6105319])

R2 = np.vstack([
                [0.9977447439, -0.02744349417, -0.06125586154],
                [0.01105126748, 0.967308141, -0.2533630398],
                [0.06620646065, 0.2521146864, 0.9654298988]
                ])
T2 = np.array([-83.91262614, -67.72431866, 370.9560732])

camera_position = np.matmul(-np.linalg.inv(R), T)
camera_position2 = np.matmul(-np.linalg.inv(R2), T2)
print(camera_position)
print(camera_position2)




# measurements = [
#     [0.5, 0.5, 0.1],
#     [1.5, 1.5, 1.5],
#     [2.7, 3.0, 2.5],
#     [3.5, 4.5, 3.5],
#     [4.2, 5.5, 4.8],
#     [5.0, 7.0, 6.0],
#     [5.8, 8.5, 7.2],
#     [6.0, 10.0, 8.0],
#     [5.5, 11.2, 7.5],
#     [4.7, 12.0, 6.8],
#     [3.5, 12.5, 6.0],
#     [2.5, 11.5, 5.0],
#     [1.5, 10.0, 4.2],
#     [0.8, 8.5, 3.5],
#     [0.3, 7.0, 3.0],
#     [0.7, 5.5, 2.5],
#     [1.5, 4.0, 2.0],
#     [2.5, 2.5, 1.5],
#     [3.0, 1.5, 1.0],
#     [3.5, 0.5, 0.1],
#     [3.5, 0, 0.1],
#     ]

# cam1 = Camera('iphonex')
# img_point = cam1.camera_to_image(measurements[0])
# # print(img_point)

# cam_point = cam1.image_to_camera(img_point)
# print(cam_point)