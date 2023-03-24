from create_video_file import VideoImgConverter
import glob
import os
import cv2


def img_to_video(filepath):
    """_summary_

    :param filepath (str path): Path to the directory of image frames
    """
    img_array = []
    count = 0 
    base_name = 'video_label_1'
    size = ()
    size_flag = False
    
    for i in range(0,1300):
        filename = f'{base_name}_{i}_*.jpg'
        file_glob = glob.glob(f'{filepath}/{filename}')
        file = ''
        
        if file_glob:
            [file] = file_glob
        
        if os.path.exists(file):
            img = cv2.imread(file)
            height, width, layers = img.shape
            
            if not size_flag:
                size = (width,height)
                size_flag = True
            img_array.append(img)
       
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    print(len(img_array))
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
# viconv = VideoImgConverter()
filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/Dataset/combined'
img_to_video(filepath)

# files = glob.glob(f'{filepath}/*.jpg')
# print(files)

# file_names = os.listdir(filepath)

# # Print each file name in the list
# for file_name in file_names:
#     print(file_name)

# sorted_files = sorted(files)
# print(sorted_files)

# count = 0
# large_count = 0
# while True:
#     if count == 0:
#         print(f'Starting count {large_count}')
#         large_count = large_count + 1
    
#     try:
#         [file] = glob.glob(f'{filepath}/video_label_1_{large_count}_*.jpg')
        
#         if os.path.exists(file):
#             # print(f'This is {file}')
#             count = count + 1
#             large_count = large_count + 1
#     except:
#         break_count = large_count + count
#         if count != 0:
#             print(f'Number of files = {count}')
#             print(f'break count {break_count}')
#         count = 0
    
#     # if os.path.exists(file):
#     #     print(f'This is {file}')
#     #     count = count + 1
#     #     large_count = large_count + 1
#     # else:
#     #     print(f'break count {count}')
#     #     count = 0
    
#     if large_count == 1300:
#         break