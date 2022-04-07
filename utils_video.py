import os
from stat import filemode
import sys
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from utils_pre import *
from shutil import copyfile,move
from utils_cv import cv_show
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from PIL import Image
from collections import deque


ImgType = ['*.jpg','*.jpeg','*.tif','*.png']
VideoType = ['*.avi','*.mp4']
LabelType = ['*.xml']


def Video2Video(videofile,savedir,interval,offset):
    video_deque = deque()
    print(videofile)
    cap = cv2.VideoCapture()
    cap.open(videofile)
    rate = cap.get(cv2.CAP_PROP_FPS)
    
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Current video fps:{}".format(rate))
    print("Current video frame No.:{}".format(totalFrameNumber))
    if offset+interval > totalFrameNumber:
        print("No video!")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(savedir)
    videoWriter = cv2.VideoWriter(savedir, fourcc, 25.0, (w,h)) 
    n = 0
    while True:
        # print("ok")
        ret, frame = cap.read()
        # video_deque.appendleft(frame)
        if ret != True:
            break 
        if ((n+offset)%interval) == 0:    

            # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
            # print(im_name)
            videoWriter.write(frame)
        else:
            pass
        n += 1
    cap.release()
    videoWriter.release()

# def Pic2Video(imgfiles,savedir,fps=25):
#     fourcc = VideoWriter_fourcc(*'mp4v')
#     image = Image.open(imgfiles[0])
#     videoWriter = cv2.VideoWriter(savedir, fourcc, fps, image.size)
#     for im in imgfiles:
#         frame = cv2.imread(im)  # 这里的路径只能是英文路径
#         # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
#         # print(im_name)
#         videoWriter.write(frame)
#     print("图片转视频结束！")
#     videoWriter.release()
#     cv2.destroyAllWindows()
    

def main_video2video(videodir):
    '''
        pic to video 
    '''
    _,filelist = getFiles(videodir,VideoType)
    interval = int(input("Input interval frame number:"))
    offset = int(input("Input offset frame number:"))
    savedir = mkFolder(videodir,'video')
    for file in filelist:
        Video2Video(os.path.join(videodir,file),os.path.join(savedir,file),interval,offset)

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"D:\02_Study\01_PaddleDetection\Pytorch\yolov5\data\images/"
        file_dir = r"D:\01_Project\01_Pangang\08_Video\test0331\falldown_alarm/"
        # pass
    if action == "getFrame":
        print(main_extract_frame_from_video.__doc__)
        main_video2video(file_dir)
    main_video2video(file_dir)
    os.system("pause")