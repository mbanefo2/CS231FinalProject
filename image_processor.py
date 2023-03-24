import cv2
import os
import numpy as np

class ImageProcessor(object):
	def __init__(self, filepath, result_path, debug=False):
		self.corners = []
		self.found_corners = False
		self.debug = debug
		self.filepath = filepath
		self.result_path = result_path
        
		if self.debug and not self.result_path:
			raise ValueError('Must provide result path')
        
		if self.result_path:
			if not os.path.isdir(self.result_path):
				os.makedirs(self.result_path)
            
		self.original_img = cv2.imread(filepath)
		self.height, self.width = self.original_img.shape[:2]
    
	def get_image_object(self):
		return cv2.imread(self.filepath)

	def dim_non_white_colors(self, img):
		# Apply a binary threshold to obtain a mask of white pixels
		_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

		# Invert the mask to obtain a mask of non-white pixels
		mask = cv2.bitwise_not(mask)

		# Apply the mask to the input image to dim all non-white pixels
		dimmed = cv2.bitwise_and(img, img, mask=mask)

		if self.debug:
			cv2.imwrite(f'{self.result_path}/dimmed.png', dimmed)

    
	def convert_to_grey(self, img):
		# img = self.get_image_object()
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	def gaussian_blur(self, img, kernel_size=5):
		blur = cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)
		if self.debug:
			cv2.imwrite(f'{self.result_path}/g_blur.png', blur)
		return blur

	def adaptive_threshold(self, img):
		thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        
		if self.debug:
			cv2.imwrite(f'{self.result_path}/thresh.png', thresh)
		return thresh
    
	def morph_open(self, img, kernel_size=2):
		kernel = np.ones((kernel_size,kernel_size), np.uint8)
		opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

		if self.debug:
			cv2.imwrite(f'{self.result_path}/morph_open.png', opening)
		return opening

	def canny_edge(self, img, thresh1=50, thresh2=100, aperture_size=5):
		edges = cv2.Canny(img, thresh1, thresh2, apertureSize=aperture_size, L2gradient=True)

		if self.debug:
			cv2.imwrite(f'{self.result_path}/edges.png', edges)

		return edges

	def dilate_img(self, img, kernel_size=2):
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		dilated_edges = cv2.dilate(img, kernel, iterations=1)

		if self.debug:
			cv2.imwrite(f'{self.result_path}/dilated_edges.png', dilated_edges)
		return dilated_edges

	def find_contours(self, img):
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		rect_contours = []
		for contour in contours:
		# Approximate the contour to reduce the number of points
			approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
		
			# If the contour has four corners, it's likely the badminton court
			if len(approx) == 4:
				# Verify that the contour is roughly rectangular in shape
				if cv2.isContourConvex(approx):
					rect_contours.append(approx)
					# new_img = cv2.drawContours(new_img, [approx], 0, (0, 0, 255), 2)

		if not rect_contours:
			raise ValueError('No rectangular contours found') 
		return rect_contours

	def find_largest_contour(self, contours):
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)
		largest_contour = contours[max_index]
		approx = cv2.approxPolyDP(largest_contour, 0.01*cv2.arcLength(largest_contour, True), True)
		return approx, largest_contour

	def get_dominant_color(self, img):
        # Initialize the result as a zero-filled NumPy array of shape (3,)
		result = np.zeros((3,), dtype=int)

    	# Define the number of bins and the bin width
		bins = 64
		bin_width = 256 // bins

    	# Loop over each color channel
		for i in range(3):
        	# Compute the color histogram for the current channel
			hist = cv2.calcHist([img], [i], None, [bins], [0, 256])

        	# Smooth the histogram by adding neighboring bin values
			hist_smoothed = np.convolve(hist.squeeze(), [1, 2, 1], mode='same')

        	# Find the index of the peak in the smoothed histogram
			peak_index = np.argmax(hist_smoothed)

        	# Compute the value of the dominant color for the current channel
			color_value = (peak_index + 0.5) * bin_width

        	# Store the value in the result array
			result[i] = color_value

		# Return the result as a NumPy array of shape (3,)
		return result

	def get_rgb_mask(self, img):
		# Find the biggest region that closely matches the court's average color in RGB space
		win_rgb = img.copy()
		win_dominant_rgb = self.get_dominant_color(win_rgb)

		# Define the color thresholds for the RGB mask
		color_thresh = 40
		lower_rgb = win_dominant_rgb - color_thresh
		upper_rgb = win_dominant_rgb + color_thresh

		# Create the RGB mask
		rgb_mask = cv2.inRange(img, lower_rgb, upper_rgb)
		if self.debug:
			cv2.imwrite(f'{self.result_path}/rgb_mask.png', rgb_mask)
       
		return rgb_mask

	def get_hsv_mask(self, img):
		# Find the biggest region that closely matches the court's average color in HSV space
		win_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		win_dominant_hsv = self.get_dominant_color(win_hsv)

		# Define the color thresholds for the HSV mask
		sat_thresh = 4
		hue_thresh = 30
		val_thresh = 1000
		lower_hsv = win_dominant_hsv - [sat_thresh, hue_thresh, val_thresh]
		upper_hsv = win_dominant_hsv + [sat_thresh, hue_thresh, val_thresh]

		# Create the HSV mask
		hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv_mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

		# Optionally save the mask to disk
		if self.debug:
			cv2.imwrite(f'{self.result_path}/hsv_mask.png', hsv_mask)

	def bitwise_mul(self, mask1, mask2):
		new_mask = cv2.bitwise_and(mask1, mask2)

		if self.debug:
			cv2.imwrite(f'{self.result_path}/new_mask.png', new_mask)
   
	def contour_from_mask(self, mask, elliptical_shaper=20):
		# Perform morphological closing to fill in small gaps in the court mask
		kernel_size = int(self.width / elliptical_shaper)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		# Optionally save the closed mask to disk
		if self.debug:
			cv2.imwrite(f'{self.result_path}/closed_mask.png', closed_mask)

		# Find the largest contour in the closed mask. This is assumed to be the court.
		contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		max_area = 0
		max_contour = None
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > max_area:
				max_area = area
				max_contour = contour
		return max_contour


