import cv2
from court_finder_class import CourtFinder

ROOT_DIR = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/Dataset/train/background_subtraction_test'

# Step 1: Collect a set of image frames that contain the badminton court from various angles.
filepath = f'{ROOT_DIR}/video_label_1_119_jpg.rf.fb6487e3c2d8212062cc1bb07b960784.jpg'
# filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/UntrackedFiles/out/rgb_mask.png'
# filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/code/lucas-kanade/test/court_edges.png'
# filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/image_frames/frame_0001.png'

result_path = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/code/court_class/debug_result'
cf = CourtFinder(filepath, result_path, True)

img = cf.get_image_object()
gray = cf.convert_to_grey(img)

# non_white = cf.dim_non_white_colors(gray)
dom_col = cf.get_dominant_color(img)
print(dom_col)

blur = cf.gaussian_blur(gray)

thresh = cf.adaptive_threshold(blur)
morph = cf.morph_open(thresh)
edge = cf.canny_edge(morph)
dilated_edge = cf.dilate_img(edge)
contours = cf.find_contours(dilated_edge)
approx, largest_contour = cf.find_largest_contour(contours)

# Draw the contour on the original image
# cv2.drawContours(img, [largest_contour], 0, (0, 0, 255), 2)

# Draw circles on the corners of the court
# for corner in largest_contour:
#     x, y = corner.ravel()
#     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# # Show the original image with the detected corners
# # cv2.imwrite("./tutor_results/new_img.png", new_img)
# cv2.imshow('Badminton Court', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()