import cv2
import numpy as np
import glob
import os

class VideoImgConverter:
    """Class to convert image frames to video files and vice versa
    """
    def __init__(self):
        self.img_array = []
    
    def img_to_video(self, filepath):
        """_summary_

        :param filepath (str path): Path to the directory of image frames
        """
        for filename in glob.glob(f'{filepath}/*.jpg'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            self.img_array.append(img)
        
        
        out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()
    
    def video_to_img(self, videopath, outpath):
        
        # Open the video file
        video_capture = cv2.VideoCapture(videopath)
        
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Initialize a counter for the frames
        frame_count = 0

        # Loop through the frames of the video
        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()

            # If the frame was not successfully read, break out of the loop
            if not ret:
                break

            # Increment the frame counter
            frame_count += 1

            # Write the frame as a PNG image file
            image_file = f'frame_{frame_count:04d}.png'
            cv2.imwrite(f'{outpath}/{image_file}', frame)

        # Release the video capture object and close all windows
        video_capture.release()
        cv2.destroyAllWindows()

conv = VideoImgConverter()
conv.video_to_img('/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/videos/IMG_3194_right__1__MOV_AdobeExpress.mp4','/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/image_frames/')