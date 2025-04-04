#!/usr/bin/env python
# coding:utf-8
from cProfile import label
import os
from stat import filemode
import sys
import json
import time#!/usr/bin/env python
# coding:utf-8
from cProfile import label
import os
from stat import filemode
import sys
import json
import time
import glob
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from utils_xml import *
from utils_math import *
from utils_cv import *
from shutil import copyfile,move
from sklearn.model_selection import train_test_split
import random
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

ImgType = ['*.jpg','*.jpeg','*.tif','*.png','*.bmp']
VideoType = ['*.avi','*.mp4','hav','.h5']
LabelType = ['*.xml']
pyscriptpath = os.path.split(os.path.realpath(__file__))[0]
configpath = os.path.join(pyscriptpath,"config.json")

with open(configpath, 'r') as c:
    config = json.load(c)

def preprocess(imgdir):
    xmldir = imgdir.replace(imgdir[-4:],".xml")
    objectlist,w,h = getObjectxml(xmldir,"all")
    img = cv2.imread(imgdir)
    return img,objectlist,w,h

def calc_mode(nums):
    '''
    Calculate mode
    :param nums: num:list
    :return: mode,mode_count
    '''
    unique_values, value_counts = np.unique(nums, return_counts=True)

    # 找到频次数组中的最大值索引
    max_count_index = np.argmax(value_counts)
    # 返回众数
    mode = unique_values[max_count_index]
    mode_count = value_counts[max_count_index]
    return mode,mode_count

def window_xml(xmlpath,bboxes,window,cls=["person"]):
    h = window[2]-window[0];w = window[3]-window[1];c=3

    new_bbox = []
    for bbox in bboxes:
        new_xmin = max(bbox[1]-window[0],0);new_ymin = max(bbox[2]-window[1],0)
        new_xmax = min(bbox[3]-window[0],w);new_ymax = min(bbox[4]-window[1],h)
        if new_xmin >= new_xmax or new_ymin >= new_ymax :
            pass
        else:
            if bbox[0] in cls:
                temp = [new_xmin,new_ymin,new_xmax,new_ymax,1,cls.index(bbox[0])]
                new_bbox.append(temp)
    xmldic = {"size":{"w":str(w),"h":str(h),"c":str(c)},"object":new_bbox}
    createObjxml(xmldic,xmlpath,cls)

def scaleBoundingBox(data,oimg,dimg):
    '''
    Description: Scale the bbox size from origin box width and height to destination。
    Author: Yujin Wang
    Date: 2022-10-10
    Args:
        data[list]:bbox list
        oimg[tuple]:Origin image shape
        dimg[tuple]:destination image shape
    Return:
        data[list]:bbox list
    Usage:
    '''
    ow,oh = oimg
    w,h = dimg
    w_scale = float(w / ow);
    h_scale = float(h / oh)
    x1,y1,x2,y2 = float(data[1]),float(data[2]),float(data[3]),float(data[4])
    data[1] = round(x1 * w_scale);
    data[2] = round(y1 * h_scale) ;
    data[3] = round(x2 * w_scale) ;
    data[4] = round(y2 * h_scale)
    return data

def findRelativeFiles(filepath):
    '''
    Description: Find these files which has the same name as the input file  
    Author: Yujin Wang
    Date: 2022-02-15
    Args:
        filepath[str]:"C:/a.txt"
    Return:
        relative files[list]
    Usage:
    '''
    filename,type = os.path.splitext(filepath)
    relativefiles = glob.glob(filename+'.*')
    return relativefiles


def writeFile(filedir,data):
    '''
    Description: Write samplesets failes in a txt 
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        filedir: txt file
        data: text
    Return:
        NaN
    Usage:
        train_files, val_files = train_test_split(jpgfiles, test_size=0.1, random_state=55)
        sampleset(train_files,"./",fn = 'train.txt')
    '''
    f = open(filedir,'w')
    for i in data:
        f.write(i)
        f.write("\n")
    f.close()

def mkFolder(dir,foldername):
    '''
    Description: Make a folder
    Author: Yujin Wang
    Date: 2022-02-13
    Args:
        dir[str]:folder directory
        foldername[str]:folder name
    Return:
        new folder directory
    Usage:
    '''
    try:
        savedir = Path(dir + foldername)
        savedir.mkdir(parents=True, exist_ok=True)
    except:
        savedir = Path(dir + "_")
        savedir.mkdir(parents=True, exist_ok=True)
    return savedir

def getFiles(folddir,typelist):
    '''
    Description: get files in folder
    Author: Yujin Wang
    Date: 2022-02-13
    Args:
        folddir[str]:  folder directory
    Return:
        files list:(path, name)
    Usage:
    '''
    files = []

    folddir = folddir + os.path.sep
    for type in typelist:
        files.extend(glob.glob(folddir + type))
    files_wodir = [Path(i.replace("\\", "/").split("/")[-1].split(".json")[0]) for i in files]
    return files,files_wodir


def renFile(filedir,savedir,format,label,id=0):
    '''
    Description:
        Rename file in filedir
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        filedir[str]:files directory
        format[str]: file format
        outfiledir[str]:output dir, if ='', replace original file
    Return:
        NaN
    Usage:
        renFile(filedir,'.jpg')
    '''
    _,files = getFiles(filedir, [format])
    total = len(files)
    for _,file in enumerate(files):
        print("%d/%d Currrent image: %s" %(id,total,file))
        str1 = file[:-3]+'*'
        duplicatefiles = glob.glob(filedir+str1)
        try:
            if label != '':
                newname = os.path.join(savedir , str(id)+ '_' + label)
            else:
                
                newname = os.path.join(savedir , str(id)+ '_' + file[:-4])

            for file in duplicatefiles:
                copyfile(file,newname + file[-4:])  
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        id += 1



def getFrame(dir,flielist,intertime=100,timeToStart = 1):
    '''
    Description: Extract frame from video
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        dir[str]: video dir.
        flielist[list]:video list
        savedir[str]: frame save directory
    Return:
        NaN
    Usage:
        avi_list =  glob.glob(DocDir+".avi")
        filelist = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in avi_list[0]]
        print (filelist)
        getFrame(avi_list[0],filelist,savedir)
    '''
    savedir = mkFolder(dir,"frame")
    num = 0
    for index,file in enumerate(flielist):
        num += 1
        cap = cv2.VideoCapture()
        print (file)
        cap.open(dir+file)
        if cap.isOpened() != True:
            os._exit(-1)
        
        #get the total numbers of frame
        totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print ("the number of frames is {}".format(totalFrameNumber))

        #get the frame rate
        rate = cap.get(cv2.CAP_PROP_FPS)
        print ("the frame rate is {} fps".format(rate))
        
        #set the start frame to read the video
        
        frameToStart = int(timeToStart * rate)+1
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
        except:
            print("Start time is gt than video total time.")
        
        # get each frames and save
        frame_num = 0 
        date = time.strftime("%Y-%m-%d", time.localtime())
        while True:
            
            ret, frame = cap.read()
            if ret != True:
                break
            frametime = round((frame_num +frameToStart-1)/ rate,2)
            if intertime == 0:
                filename = str(frame_num) + "_" +file[:-4] + "_" + str(frametime).replace('.','p') + ".jpg"
                img_path = os.path.join(savedir ,filename)
                # print (img_path)
                cv2.imwrite(img_path,frame)
            else:    
                if frame_num % (intertime*rate) == 0:
                    filename = file[:-4] + "_" + str(frametime).replace('.','p') + ".jpg"
                    img_path = os.path.join(savedir ,filename)
                    print (img_path)
                    cv2.imwrite(img_path,frame)
            frame_num += 1
        
            # wait 10 ms and if get 'q' from keyboard  break the circle
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()


def Yolo2VOC(imgfiles,classes):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args:
        yolofile[str]: .xml file from labelimg
        classes[list]: Class name
    Return:
        obj[list]: obeject list,[['person', 592, 657, 726, 1077],['person', 592, 657, 726, 1077]]
    Usage:
        bboxlist = getObjectxml(yolofile,classes)
    '''

    # print ("Current process file:",yolofile)
    total = len(imgfiles)
    id = 1
    for imgfile in imgfiles:
        try:
            print("%d/%d Currrent image: %s" % (id, total, imgfile))
            im = cv2.imread(imgfile)
            im_h,im_w,im_c = im.shape
            bbox = []
            id += 1

            for line in open(imgfile[:-4]+'.txt'):

                clsid,cx,cy,w,h = [float(i) for i in line.split()]
                xmin,ymin,xmax,ymax =  xywh2xyxy([im_h,im_w],[cx,cy,w,h])
                bbox.append([xmin,ymin,xmax,ymax,0,clsid])
            xmldic = {"size": {"w": str(im_w), "h": str(im_h), "c": str(im_c)}, "object": bbox}
            createObjxml(xmldic, imgfile, cls=classes, xmlfile=None)

        except Exception as e:
            print(e)
            print("Error:imgfile:",imgfile)


def VOC2Yolo(xmlfiles,classes='all'):
    '''
    Description: Change xml to yolo format
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        xmlfiles[list]: xml files
        classes[list]: save classes 
    Return:
        NaN
    Usage:
        xmlfiles = glob.glob("./annotations" + '*.xml')
        classes = ["mask","nomask"]
        VOC2Yolo(xmlfiles,classes)
    '''
    total = len(xmlfiles)
    id = 1
    for file in xmlfiles :
        file = file.replace("\\", "/")
        # a = cv2.imread(file.replace(".xml",".jpg"))
        print("%d/%d Currrent image: %s" %(id,total,file))
        out_file = open(file.replace(file[-4:],".txt"),'w') 
        bboxlist,width,height = getObjectxml(file,classes)
        for bbox in bboxlist:
            try:
                cls_id = classes.index(bbox[0])
                b = (float(bbox[1]), float(bbox[3]), float(bbox[2]), float(bbox[4]))
                bb = xyxy2xywh((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            except:
                print("No object found in xml, file:%s" %(file))
        id += 1
        out_file.close()

def sampleset_paddle(filelist,dir,fn = "train.txt"):
    '''
    Description: Write samplesets failes in a txt 
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        filelist[list]:xml file list
        dir[str]: Directory of txt file
    Return:
        NaN
    Usage:
        train_files, val_files = train_test_split(jpgfiles, test_size=0.1, random_state=55)
        sampleset_paddle(train_files,"./",fn = 'train.txt')
    '''
    savedir = dir
    try:
        os.mkdir(savedir)
    except:
        pass
    fd = dir +'/'+ fn
    f = open(fd,'w')
    for i in filelist:
        line = i + " "+i.replace(".jpg",".xml") +"\n"
        f.write(line)
    f.close()
    print(fn + "is Done")
 
def savecopy(filelist,file2dir):
    '''
    Description: Save files as files with index.
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        filelist[list]:files
        file2dir[str]: files directory
    Return:
        NaN
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'    
        xmlfiles = glob.glob(filedir + '*.xml')
        savecopy(xmlfiles,filedir+"copy/")
    '''
    savedir = file2dir
    try:
        os.mkdir(savedir)
    except:
        pass
    index = 0
    for file1 in filelist:
        file2xml = file2dir + str(index) + '.xml'
        file2jpg = file2dir + str(index) + '.jpg'
        copyfile(file1,file2xml)
        copyfile(file1.replace('.xml','.jpg'),file2jpg)
        index += 1

def getImgMaxLongEdge(imgpath):
    '''
    Description: Get image shape information.
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        imgpath[str]:image path
    Return:
        h,w,h//w,h%w,'w'[tupel]:
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'
        xmlfiles = glob.glob(filedir + '*.xml')
        savecopy(xmlfiles,filedir+"copy/")
    '''

    img = cv2.imread(imgpath)
    (h,w,_) = img.shape
    if h>w:
        return h,w,h//w,h%w,'w'
    if h<=w:
        return w,h,w//h,w%h,'v'

def createSquarImg(imgfiles,pob=1,flip = ['v','h','vh',"o"]):
    '''
    Description: Creat square image
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        imgfiles[list]:images' path list
        pob[float]:probability
        flip[list]:flip style list
    Return:
        img[img]
    Usage:

    '''
    maxedge, minedge,ratio, padding, direction = getImgMaxLongEdge(imgfiles[0])
    num = len(imgfiles[1:])
    frame_padding = (len(imgfiles[1:])-num) /2.
    if direction == "v":
        padding = np.ones([int(padding/(num)), maxedge, 3], dtype=np.uint8)
    else:
        padding = np.ones([maxedge, int(padding / (num)), 3], dtype=np.uint8) 
    padding[:,:,0] = 255
    img = cv2.imread(imgfiles[0])

    for id,img1 in enumerate(imgfiles[1:]):
        img1 = cv2.imread(img1)
        if flip[id] == "o":
            pass
        elif flip[id] == "hv":
            img1,_ = reflectimg(img1, prob=pob, fliptype='h')
            img1,_ = reflectimg(img1, prob=pob, fliptype='v')
        else:
            img1,_ = reflectimg(img1, prob=pob, fliptype=flip[id])
        # print(flip[id],img1.shape)

        if direction == "v":
            img = np.concatenate([img, padding], axis=0)
            img = np.concatenate([img, img1], axis=0)

        else:
            img = np.concatenate([img, padding], axis=1)
            img = np.concatenate([img, img1], axis=1)

    if frame_padding != 0:
        if direction == "v":
            padding = np.ones([minedge,maxedge,3], dtype=np.uint8) * 255
            img = np.concatenate([img, padding], axis=0)
            img = np.concatenate([padding,img], axis=0)
        else:
            padding = np.ones([maxedge,minedge,3], dtype=np.uint8) * 255
            img = np.concatenate([img, padding], axis=1)
            img = np.concatenate([padding, img], axis=1)

    return img

def paddingSquare(img):
    '''
    Description: add padding in image to a square image
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        img[image]
    Return:
        img[img]
    Usage:

    '''
    height, width, _ = img.shape
    if height > width:
        padding = np.ones([height, int((height - width) / 2), 3], dtype=np.uint8) * 0
        img = np.concatenate([padding, img], axis=1)
        img = np.concatenate([img, padding], axis=1)
    else:
        padding = np.ones([int((width - height) / 2), width, 3], dtype=np.uint8) * 0
        img = np.concatenate([padding, img], axis=0)
        img = np.concatenate([img, padding], axis=0)
    pad_height, pad_width, _ = img.shape
    return img,int((pad_height - height)/2),int((pad_width - width)/2)


def saveCropImgcopy(imgdir,imgfile,clsname,scale=0.1,square = True):
    '''
    Description: Crop image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
        saveimgfile: croped image is saved in dir and file name
    Return:
    Usage:
    '''

    savedir = mkFolder(imgdir,clsname +"_" +'crop')
    xmlfile = imgfile.replace(imgfile[-4:],".xml")
    objectlist,w,h = getObjectxml(imgdir + xmlfile,[clsname])
    img = cv2.imread(imgdir +imgfile)

    img,hoffset,woffset = paddingSquare(img)
    height, width, _ = img.shape

    id = 0
    if len(objectlist) > 0 and objectlist[0]!=[]:
        for object in objectlist:
            id += 1
            xmin = int(object[1])+woffset;
            ymin = int(object[2])+hoffset;
            xmax = int(object[3])+woffset;
            ymax = int(object[4])+hoffset
            h = ymax - ymin; w = xmax - xmin
            scale1 = int((max(w,h)*(scale+1)-max(w,h))/2)
            offset = int(abs((h - w) / 2))
            confidence = object[5] if object[5]!= 0 else 0
            if square == True:
                if h > w:
                    y1 = ymin - scale1 ;y2 = ymax + scale1 ;x1 = xmin - offset - scale1 ; x2 = xmax + offset + scale1
                    object = [[scale1 + offset, scale1  , w + offset + scale1 , h + scale1 , confidence,0 ]]

                else:
                    y1 = ymin - offset - scale1 ;y2 = ymax + offset + scale1 ; x1 = xmin-scale1 ;x2 = xmax+scale1
                    object = [[scale1 , scale1 +offset, w+scale1 , h+offset+scale1 , confidence, 0]]
            else:
                y1 = ymin-scale1 ; y2 = ymax+scale1 ; x1 = xmin-scale1 ; x2 = dxmax+scale1
                object = [[0,0,w,h, confidence, 0]]

            ymin = max(0, y1);ymax = min(y2, height); xmin = max(0, x1);xmax = min(x2, width)
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w, c = crop_img.shape
            xmldic = {"size": {"w": str(w), "h": str(h), "c": str(c)},
                      "object": object}
            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id)+'.jpg')
            # h,w,c = crop_img.shape

            createObjxml(xmldic,saveimg,cls=[clsname])
            cv2.imwrite(saveimg,crop_img)



def saveSquareImg(imgdir, imgfile, clsname):
    savedir = mkFolder(imgdir, clsname + "_" + 'crop')
    imgfulldir = imgdir + imgfile
    img = cv2.imread(imgfulldir)
    height, width, _ = img.shape
    edgesize = min(height,width)
    xmlfile = imgfulldir.replace(imgfulldir[-4:], ".xml")
    objectlist, w, h = getObjectxml(xmlfile, [clsname])
    id = 0
    if len(objectlist) > 0 and objectlist[0] != []:
        for objectbox in objectlist:
            id += 1
            xmin = int(objectbox[1]) ;
            ymin = int(objectbox[2]) ;
            xmax = int(objectbox[3]) ;
            ymax = int(objectbox[4]) ;
            h = ymax - ymin;
            w = xmax - xmin
            if xmin == 0 :
                xmax = edgesize
            if xmax == width:
                xmin = width - edgesize;

            if ymin == 0:
                ymax = edgesize
            if ymax == height:
                ymin = height-edgesize ;
            crop_img = img[ymin:ymax, xmin:xmax]
            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id) + '.jpg')
            cv2.imwrite(saveimg, crop_img)




def saveCropImg(imgdir, imgfile, clsname, scale=0.1, square=True,resize_img =0,fixroi = False):
    '''
    Description: Crop image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args:
        img[image]: image
        object: dest object
        saveimgfile: croped image is saved in dir and file name
    Return:
    Usage:
    '''
    def box_in_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """
        判断box (x1, y1, x2, y2)是否在box (xmin, ymin, xmax, ymax) 之中
        """
        xc = (x1+x2)/2;yc = (y1+y2)/2
        return (xmin <= xc <= xmax) and (ymin <= yc <= ymax),(xmin <= x1 <= xmax) and (xmin <= x2 <= xmax) and (ymin <= y1 <= ymax) and (ymin <= y2 <= ymax)
    savedir = mkFolder(imgdir, clsname + "_" + 'crop')
    imgfulldir = imgdir + imgfile
    img = cv2.imread(imgfulldir)
    height, width, _ = img.shape
    if fixroi == False:
        xmlfile = imgfulldir.replace(imgfulldir[-4:], ".xml")
        objectlist, w, h = getObjectxml(xmlfile, [clsname])
    else:
        try:
            xmlfile = imgdir + "crop.xml"
            objectlist, w, h = getObjectxml(xmlfile, 'all')
        except Exception as e:
            print(e)
            print(traceback.format_exc())


    img, hoffset, woffset = paddingSquare(img)
    id = 0
    if len(objectlist) > 0 and objectlist[0] != []:
        for objectbox in objectlist:
            id += 1
            xmin = int(objectbox[1]) + woffset;
            ymin = int(objectbox[2]) + hoffset;
            xmax = int(objectbox[3]) + woffset;
            ymax = int(objectbox[4]) + hoffset
            h = ymax - ymin;
            w = xmax - xmin
            scale1 = int((max(w, h) * (scale + 1) - max(w, h)) / 2)
            offset = int(abs((h - w) / 2))
            confidence = objectbox[5] if objectbox[5] != 0 else 0
            if square == True:
                if h > w:
                    y1 = ymin - scale1;
                    y2 = ymax + scale1;
                    x1 = xmin - offset - scale1;
                    x2 = xmax + offset + scale1
                    object = [[scale1 + offset, scale1, w + offset + scale1, h + scale1, confidence, 0]]

                else:
                    y1 = ymin - offset - scale1;
                    y2 = ymax + offset + scale1;
                    x1 = xmin - scale1;
                    x2 = xmax + scale1
                    object = [[scale1, scale1 + offset, w + scale1, h + offset + scale1, confidence, 0]]
            else:
                y1 = ymin - scale1;
                y2 = ymax + scale1;
                x1 = xmin - scale1;
                x2 = xmax + scale1
                object = [[0, 0, w, h, confidence, 0]]

            ymin = max(0, y1);
            ymax = min(y2, height);
            xmin = max(0, x1);
            xmax = min(x2, width)
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w, c = crop_img.shape

            classes = [clsname]
            innerobjectlist, _, _ = getObjectxml(xmlfile, "all")
            for innerbox in innerobjectlist:
                confidence = innerbox[5] if innerbox[5] != 0 else 0
                centerinflag,_ = box_in_box(innerbox[1], innerbox[2], innerbox[3], innerbox[4], objectbox[1], objectbox[2], objectbox[3],objectbox[4])
                if innerbox[0] != clsname and centerinflag:
                    if innerbox[0] not in classes:
                        classes.append(innerbox[0])
                    x1 = object[0][0] + innerbox[1]-objectbox[1]; y1 = object[0][1] + innerbox[2]- objectbox[2] ;
                    object.append([x1, y1, x1 + innerbox[3] - innerbox[1],y1 + innerbox[4] - innerbox[2], confidence ,classes.index(innerbox[0])])


            if resize_img !=0 :
                crop_img = cv2.resize(crop_img,(resize_img ,resize_img ))
            else:
                xmldic = {"size": {"w": str(w), "h": str(h), "c": str(c)},"object": object}
                # createObjxml(xmldic, saveimg, cls=classes)

            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id) + '.jpg')
            cv2.imwrite(saveimg, crop_img)




def plotRectBox(img,objectlist,names):
    '''
    Description: Plot bndbox and label in image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
    Return:
        img
    Usage:
    '''
    sys.path.append(config["yolov5"])
    from utils.plots import Annotator, colors
    annotator = Annotator(img, line_width=1, example="")
    
    for object in objectlist:
        if len(object)==6:
            label, xmin, ymin, xmax, ymax,conf =object[0],object[1], object[2], object[3], object[4],object[5]
        else:
            label, xmin, ymin, xmax, ymax,conf =object[0],object[1], object[2], object[3], object[4],0
        c = names.index(label)
        label =  f'{names[c]} ' #{conf:.2f}
        annotator.box_label([xmin, ymin, xmax, ymax], label, color=colors(c, True))
        # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        # cv2.putText(img, object[0], (int((xmin+xmax)/2),int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    return annotator.result()
    # cv_show('img',img)

def changeHSV1(img,adjh=1.0,adjs=1.0,adjv=1.1):
    # print(adjh,adjs,adjv)
    adjh = adjh+0.5;adjs =adjs+0.8;adjv = adjv+1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    
    x = np.arange(0, 256)
    
    lut_hue = ((x * adjh) % 180).astype(dtype)
    lut_sat = np.clip(x * adjs, 0, 255).astype(dtype)
    lut_val = np.clip(x * adjv, 0, 255).astype(dtype)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(im_hsv,cv2.COLOR_HSV2BGR)

def changeHSV(img):
    '''
    Description:
        Change image light
    Author: Yujin Wang
    Date: 2022-02-22
    Args:
        img[cv.img]:Gray
    Return:

    Usage:
    '''
    # print(adjh,adjs,adjv)
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    # h_norm = (hue/np.max(hue)*180).astype(dtype)
    # s_norm = (sat/np.max(sat)*255).astype(dtype)
    v_norm = (val/np.max(val)*255).astype(dtype)
    img_hsv =  cv2.merge([hue,sat,v_norm])
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


def plotFigArray(imglist:list,imgshape=(0,0)):
    '''
    Description: 
          Plot multi imgs in a figure
    Author: Yujin Wang
    Date: 2022-02-21
    Args:
        filerir[str]:file directory
    Return:
        figures
    Usage:
    '''
    height, width, _ = imglist[0].shape
    num = len(imglist)
    row,col = calcImgArray(width,height,num)
    print(row,col)
    canvas = np.ones([row*height, col*width,3], dtype=np.uint8) * 255
    total = 0
    for i in range(row):
        for j in range(col):
            if total < num:
                canvas[height*i:height*(i+1),width*j:width*(j+1)] = imglist[total]
                total += 1
            else:
                break
    
    if imgshape !=(0,0):
        canvas = cv2.resize(canvas,imgshape)
    # cv_show("canvas", canvas)
    return canvas

def Video2Video(videofile,savedir,interval,offset,scale):
    '''
    Description: Reduce frame numbers of video or reshape video size
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        videofile[str]:video path
        savedir[str]: new video path
        interval[int]: video interval number
        offset[int]: number of frames
        scale[float]:scale of video shape
    Return:
        NAN
    Usage:

    '''
    print(videofile)
    cap = cv2.VideoCapture()
    cap.open(videofile)
    rate = cap.get(cv2.CAP_PROP_FPS)
    
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Current video fps:{}".format(rate))
    print("Current video frame No.:{}".format(totalFrameNumber))
    if offset+interval > totalFrameNumber:
        print("offset+interval > totalFrameNumber")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    print(w,h)
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
            videoWriter.write(frame)
        else:
            pass
        n += 1
    cap.release()
    videoWriter.release()


# MAIN PROCESSE
def main_extract_frame_from_video(videodir):
    '''
        Extract frame from video
    '''
    _,filelist = getFiles(videodir,VideoType)
    # startid = int(input("Start image ID:"))
    interval = float(input("Interval time(s):"))
    OffsetTime = float(input("Offset time(s):"))
    getFrame(videodir,filelist,interval,OffsetTime)



def main_remove_obj_from_xml(xmldir,cls1 = "zdxtp,zgxtp,zh,kd,yh"): 
    '''
        Remove object from xml
    '''
    _,xmlfiles = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    if cls =='N':
        cls = cls1
    cls = cls.split(',')
    print(f"Classes : f{cls}")
    remObjectxml(xmldir,xmlfiles,cls,isSave=False)
 


def main_change_voc_to_yolo(xmldir,cls=[]):
    '''
        Change VOC to Yolo
    '''
    xmlfiles, _ = getFiles(xmldir, LabelType)
    if cls==[]:
        cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
        cls_name = cls_name.split(',')
    else:
        cls_name = cls
    if type(cls) != list:
        print('Input is not correct')
    VOC2Yolo(xmlfiles,cls_name)

def main_change_yolo_to_voc(imgdir):
    '''
        Change  Yolo to Voc
    '''
    imgfiles,_ = getFiles(imgdir,ImgType)
    cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
    cls_name = cls_name.split(',')
    Yolo2VOC(imgfiles,cls_name)

def main_change_cls_name(xmldir,oldcls1 = "Hs_DTDW,Hs_BS1,Hs_ZP,Hs_DQP1,Hs_ZDXTP1,Hs_BY,Hs_JZ,Hs_QP,Hs_DR,Hs_KD1,Hs_HTP,Hs_YH,Hs_DZYR,Hs_JQTW,Hs_HS,Hs_HX,Hs_CS,Hs_BDXTP,Hs_KD,Hs_BS,Hs_ZH,Hs_WDXTP,Hs_BBQP,Hs_BX,Hs_TPH,Hs_ZGXTP,Hs_HB", \
     newcls1 = "dtdw,bs,zp,dqp,zdxtp,by,jz,qp,dr,kd,htp,yh,dzyr,jqtw,hs,hx,cs,bdxtp,kd,bs,zh,wdxtp,bbqp,bx,tph,zgxtp,hb"):
    '''
        Change class name
    '''
    xmlfiles = glob.glob(xmldir+ '*.xml')
    oldcls = input("Old class:")
    newcls = input("New class:")
    if oldcls == "" and newcls =="":
        oldcls == oldcls1;newcls == newcls1
        chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSave=False)
    else:
        chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSave=False)


def plothist(data,title,imgfile,bins = [10, 20, 30, 40, 50, 70],datarange=(0,1),show= False):
    nt, _, _ = plt.hist(data, bins=51, rwidth=0.5, range=datarange, align='mid')
    plt.plot([np.mean(data), np.mean(data)], [0, np.max(nt)], ":", label="Mean")
    plt.plot([np.median(data), np.median(data)], [0, np.max(nt)], "--", label="Median")
    for i in bins:
        value = np.percentile(data, i)
        plt.plot([value, value], [0, np.max(nt)], "--", label=f"{i}%")
        plt.text(value, np.max(nt), f'{round(value, 2)}', fontsize=8, rotation=90)
    plt.xticks()
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(imgfile)
    if show:
        plt.show()
    plt.close()

def main_check_label_xml(xmldir):
    '''
        Check label xml and present histgram of confidence
    '''
    xmlfiles,_ = getFiles(xmldir, LabelType)
    noobjectfiles,cls = checkLabexml(xmlfiles)

    if len(noobjectfiles) != 0:
        savedir = mkFolder(xmldir,"noobject")
        for file in noobjectfiles:
            for cpfile in findRelativeFiles(file):
                move(cpfile,savedir)
    else:
        print("No unlabeled img found!")
    
    savedir = mkFolder(xmldir, "checkres")
    clsname =",".join(list(cls.keys()))
    print(clsname)
    for name in cls.keys():
        temp = np.array(cls[name]["confidence"])
        plothist(temp,name,savedir / f'{name}_confidence.jpg')
        np.savetxt(savedir / f'{name}_confidence_{len(cls[name]["confidence"])}.csv', np.array(cls[name]["confidence"]), delimiter=",")



def main_change_file_name(xmldir):
    '''
        Rename files
    '''
    # xmlfiles = glob.glob(xmldir+ '*.xml')
    format = '*' + input("Input file format('.jpg'):")
    label = input("Add string in file name:")
    id = int(input("Start number:"))
    savedir =mkFolder(xmldir,'rename_files')
    renFile(xmldir,savedir,format,label,id)

def main_yolo_train_val_set(imgdir,task = 'test'):
    '''
        Split train and val dataset
    '''
    mvfolder = input("Do you want to move figeures to train an val folder?(Y/N)")
    if mvfolder == "Y":
        trainFolder = mkFolder(imgdir,"train")
        valFolder = mkFolder(imgdir,"validation")

    if task != 'test':
        _, Imagefiles = getFiles(imgdir, ImgType)
        img_serverdir = input("Train and validation img in serverdir(data/.../):")
        # imgfiles_serve = [img_serverdir + i for i in imgfiles]
        samplerdir = mkFolder(imgdir, 'train_val')
        test_size = float(input("Input the ratio of val:"))
        
        
        train_files, val_files = train_test_split( Imagefiles, test_size=test_size, random_state=55)
        if mvfolder == "Y":
            for imgfile in train_files:
                for file in findRelativeFiles(os.path.join(imgdir,imgfile)):
                    move(file, trainFolder)
            for imgfile in val_files:
                for file in findRelativeFiles(os.path.join(imgdir,imgfile)):
                    move(file, valFolder)
        
        
        if  test_size  == "0":
            writeFile(samplerdir / 'test.txt', imgfiles)
            return
        
        print(img_serverdir) 
        train_files =   [str(img_serverdir)+ 'train/' + str(i) for i in train_files]
        val_files =   [str(img_serverdir) +'validation/' + str(i) for i in val_files]
        writeFile(samplerdir / 'train.txt', train_files)
        writeFile(samplerdir / 'val.txt',val_files)
    else:
        imgfiles, _ = getFiles(imgdir, ImgType)
        writeFile(imgdir + '/test.txt', imgfiles)
        return

def main_imagesize_filter(imgdir):
    filelist,_ = getFiles(imgdir,ImgType)
    imgsizelist = []
    remdir = mkFolder(imgdir,"rem")
    resdir = mkFolder(imgdir,"res")
    errdir = mkFolder(imgdir, "err")
    total = len(filelist)
    for id,file in enumerate(filelist):
        print(f'{id+1}/{total}:{file}')
        try:
            img = cv2.imread(file)
            w,h,_= img.shape
            print(f'Done')
            imgsizelist.append([file,w,h,w*h])
        except:
            move(file,errdir)

    imgsizelist = pd.DataFrame(imgsizelist,columns=["file","W","H","A"])
    imgsizelist.to_csv(resdir / "imgsize.csv")
    plothist(imgsizelist["A"], "Img_Area", resdir / "A_histgram.jpg",datarange=(min(imgsizelist["A"]),max(imgsizelist["A"])),show=True)
    lowerthr = int(input("Image lower threshold(thrXthr):"))
    upperthr = int(input("Image upper threshold(thrXthr):"))
    remimg = imgsizelist[imgsizelist["A"]<upperthr][imgsizelist["A"]>lowerthr]
    for img in remimg["file"]:
        for file in findRelativeFiles(img):
            move(file,remdir)


# def expandcropimg(image, rect, expand_ratio=1):
#     '''
#     按照一定比例(expand_ratio)将rect放大后进行裁剪
#     Author:Zhangzhe
#     '''
#     imgh, imgw, c = image.shape
#     xmin, ymin, xmax, ymax = [int(x) for x in rect]
#     org_rect = [xmin, ymin, xmax, ymax]
#     h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
#     w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
#     # if h_half > w_half * 4 / 3:
#     #     w_half = h_half * 0.75
#     center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
#     ymin = max(0, int(center[0] - h_half))
#     ymax = min(imgh - 1, int(center[0] + h_half))
#     xmin = max(0, int(center[1] - w_half))
#     xmax = min(imgw - 1, int(center[1] + w_half))
#     return image[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect

def main_crop_object_img(imgdir):
    '''
        Crop objet image(include inner objects in box)
    '''
    clsname = input("Input class name:")
    try:
        scale =  float(input("Input expand ratio (max(h,w),def.1):"))
    except:
        print("Scale def. 1")
        scale =1.0
    square = True if input("Crop image with padding(Y/N),def.Y:") == "Y" else False
    try:
         resize_img = int(input("Resize(0:no):"))
    except:
        print("Scale def. 0")
        resize_img = 0.0

    clsname = clsname.split(',')
    _,imgfiles = getFiles(imgdir,ImgType)


    for file in tqdm(imgfiles):
        try:
            for cls in clsname:
                saveCropImg(imgdir,file,cls,scale,square,resize_img)
        except Exception as e:
            print(e)
            print(traceback.format_exc())



def main_plot_bbox(imgdir):
    '''
        Plot bbox in img
    '''
    savedir = mkFolder(imgdir,"plotbbox")
    _,imgfiles = getFiles(imgdir,ImgType)
    cls = input("Class you want to plot(e.g. person,mask): ")
    cls = cls.split(",")
    total = len(imgfiles)

    # imgfiles.sort(key=lambda x: int(x.split('_')[0]))

    # 更改工作目录到图片所在目录
    os.chdir(os.path.dirname(imgdir))

    # 使用OpenCV读取图片

    for id,file in enumerate(tqdm(imgfiles)):
        print(file)
        xmlfile = imgdir + file.replace(file[-4:],".xml")
        bbox,_,_ = getObjectxml(xmlfile,cls)

        img = cv2.imread(file)
        h,w,  _ = img.shape
        print(f'w:{w} h:{h}')
        img = plotRectBox(img,bbox,names=cls)
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        imgpath = "plotbbox/" + file[:-4]+".jpg"
        print(imgpath)
        # img = plot_line(img,ptStart = (1160, 110),ptEnd = (0, 630))
        # img = plot_line(img, ptStart=(960, 35), ptEnd=(0, 339))
        cv2.imwrite(str(imgpath),img)
        if id == 0:
            path = os.path.join(savedir,'video.mp4')
            print(path)
            vid_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'acv1'), 25, (int(w), int(h)))
        vid_writer.write(img)
    vid_writer.release()
    return

def plot_line(img,ptStart = (60, 60),ptEnd = (260, 260),point_color = (0,255,255)):
    thickness = 3
    lineType = 4
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    return img


def main_create_square_image_samples_one_pic(filedir1):
    '''
        Create a square image with padding
    '''
    savedir = mkFolder(filedir1,'new_dataset')
    imgfiles1,_ = getFiles(filedir1,ImgType)
    total = len(imgfiles1)
    for id,file in enumerate(imgfiles1):

        print("%d/%d Current process file: %s" %(id+1,total,file))
        imgfilescopy = imgfiles1.copy()
        imgfilescopy.remove(file)
        imgfilepath = os.path.join(savedir, os.path.split(file)[-1][:-4] + '_square' + os.path.split(file)[-1][-4:])
        edge,minedge,fignum,padding,direction = getImgMaxLongEdge(file)
        concimgs = [file]
        concimgs.extend([file for i in range(fignum-1)])
        flip = ['v', 'h', 'vh', "o"]
        img = createSquarImg(concimgs,flip = flip)
        cv2.imwrite(imgfilepath,img)
        try:
            xmlfile = [file.replace(file[-4:],".xml") for file in concimgs]
            xmlfile = combineXMLinDirection(xmlfile,edge,fignum,padding,direction,flip = flip)
            xmlfile.write(imgfilepath.replace(file[-4:],'.xml'))
        except:
            print("No xml file is found!")



def main_create_square_image_samples(filedir1):
    '''
        Create a square image with padding
    '''
    savedir = mkFolder(filedir1,'new_dataset')
    imgfiles1,_ = getFiles(filedir1,ImgType)
    filedir2 = input("Please input another dataset(for sample balance):")
    id = int(input("Please input start id:"))
    imgfiles2,_ = getFiles(filedir2,ImgType)
    total = len(imgfiles1)
    for file in imgfiles1:

        print("%d/%d Current process file: %s" %(id,total,file))
        imgfilescopy = imgfiles1.copy()
        imgfilescopy.remove(file)
        edge,fignum,padding,direction = getImgMaxLongEdge(file)
        concimgs = [file]
        if imgfiles2:
            concimgs.extend([random.sample(imgfiles2,1)[0] for i in range(fignum-1)])
        else:
            imgfiles2 =  imgfiles1.copy()
            concimgs.extend([random.sample(imgfiles2,1)[0] for i in range(fignum-1)])
        img = createSquarImg(concimgs)
        xmlfile = [file.replace(file[-4:],"xml") for file in concimgs]
        xmlfile = combineXMLinDirection(xmlfile,edge,fignum,padding,direction)
        imgfilepath = savedir,str(id) / '.tif'
        cv2.imwrite(imgfilepath,img)
        xmlfile.write(imgfilepath.replace(file[-4:],'.xml'))
        id += 1

def main_plot_infer_res(filedir): 
    '''
        Plot infer results from multi figures
    '''
    savedir = mkFolder(filedir,'infer_res_compare')
    _,filenamelist = getFiles(filedir,ImgType)
    compareflag = True
    resdirlist = []
    while compareflag:
        resdir = input("Inference another res dir('Enter over'):")
        if resdir == "":
            break
        resdirlist.append(resdir)
        
    total = len(filenamelist)

    for id,file in enumerate(filenamelist):
        imglist = [cv2.imread(os.path.join(filedir,file))]
        for resdir in resdirlist:
            filepath = os.path.join(resdir,file)
            if os.path.exists(filepath):

                imglist.append(cv2.imread(filepath))
            else:
                print("No result found! Image path: %s" %(filepath))
        
        img = plotFigArray(imglist)
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        cv2.imwrite(os.path.join(str(savedir),"res_"+file),img)

def main_change_hsv(filedir):
    '''
        Change the light
    '''
    savedir = mkFolder(filedir,'gray')
    _,filenamelist = getFiles(filedir,ImgType)
    total = len(filenamelist)
    for id,file in enumerate(filenamelist):
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        img = cv2.imread(os.path.join(filedir,file))
        img = changeHSV(img)
        cv2.imwrite(os.path.join(str(savedir),file),img)

def sobelx(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转为灰度图
    dst = np.zeros_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv_show('ori', img)
    # cv_show('res', dst)
    img_sobel = cv2.Sobel(dst,cv2.CV_8U,1,0)
    # res = cv2.cvtColor(img_sobel,cv2.COLOR_GRAY2RGB)
    # cv_show('res',img_sobel)
    return img_sobel

def main_change_sobelx(filedir):
    '''
        Change the light
    '''
    savedir = mkFolder(filedir,'gray')
    _,filenamelist = getFiles(filedir,ImgType)
    total = len(filenamelist)
    for id,file in enumerate(filenamelist):
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        img = cv2.imread(os.path.join(filedir,file))
        img = sobelx(img)
        cv2.imwrite(os.path.join(str(savedir),file),img)


def main_video2video(videodir):
    '''
        pic to video 
    '''
    _,filelist = getFiles(videodir,VideoType)
    interval = int(input("Input interval frame number:"))
    offset = int(input("Input offset frame number:"))
    scale = float(input("Input scale ratio:"))
    savedir = mkFolder(videodir,'video')
    for file in filelist:
        Video2Video(os.path.join(videodir,file),os.path.join(savedir,file),interval,offset,scale)

def main_movobject(xmldir,autoflag="", numclass=""):
    '''
        Move file included object to object dir  
    '''
    xmlfiles,_ = getFiles(xmldir,LabelType)
    if autoflag == "":
        cls = input("Class name(label list,e.g. [person,mask]):")
    else:
        cls = autoflag
    cls = cls.split(',')

    if numclass == "":
        try:
            numclass = int(input("Number threshold of labels(def. 99):"))
        except:
            print('Default number 99 will be used!')
            numclass = 99
    for i in cls:
        xmlfiles,_ = getFiles(xmldir,LabelType)
        savedir = mkFolder(xmldir,i)
        movObjectxml(xmlfiles,i,savedir,numclass)

def main_remunusedfile(xmldir):
    '''
        Remove unused files
        label：lab；image：img
    '''
    filetype = input("Image or label will be removed:")
    if filetype == "lab":
        files,_ = getFiles(xmldir,LabelType)
    elif filetype == "img":  
        files,_ = getFiles(xmldir,ImgType)
    unuseddir = mkFolder(xmldir,'unused')
    for file in files:
        print(file)
        if len(findRelativeFiles(file)) == 1:
            move(file,unuseddir)


def main_imgchangetojpg(imgdir):
    " Change images' format to jpg "
    # gencomplexfigure(imgdir)
    # gencircle(imgdir)
    imgsdir,_ = getFiles(imgdir,ImgType)
    # quality_index = input("Please input quarlity index(0~100):")
    filetype = input("You want to change to File type('.tif'):")
    ct = []
    imgsize = []
    for img in tqdm(imgsdir):
        im = cv2.imread(img)
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        imgname = img.split('.')[0] + filetype
        t0 = time.time()
        cv2.imwrite(imgname,im,params)

        ct.append(time.time()-t0)
        imgsize.append([imgname,os.path.getsize(imgname)])
        print(f"{imgname},{os.path.getsize(imgname)}")
    ct = np.array(ct);imgsize = np.array(imgsize);
    ct_m,ct_std = np.mean(ct)*1000, np.std(ct)*1000
    print("%5.3f\t%5.3f" %(ct_m,ct_std))
    return ct_m,ct_std,imgsize

def gencomplexfigure(dir):
    for j in range(8):
        h = np.power(2,(j+1))
        n =int( 320/ h)
        block0 = np.zeros(( h,h, 3), dtype=np.uint8)
        block1 = np.zeros(( h,h, 3), dtype=np.uint8)+255
        img0 = np.hstack((block0, block1))
        img1 = np.hstack((block1, block0))
        img2 = np.vstack((img0, img1))
        row = img2
        for i in range(n-1):
            row = np.hstack((row,img2))
        img = row
        for i in range(n-1):
            img = np.vstack((img, row))
        params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        cv2.imwrite(f"{dir}{j}.bmp", img)


def gencircle(dir):
    '''

    Args:
        dir: Save dir：string

    Returns:

    '''

    height, width = 640, 640
    background = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    for j in range(8):
        radius = np.power(2, (j + 1))
        color = (255, 255, 255)  # 白色
        thickness = -1  # 填充圆形的内部
        cv2.circle(background, (center_x, center_y), radius, color, thickness)
        cv2.imwrite(f"{dir}{j}.bmp", background)

def main_split_images(imgdir):
    " Split image to several jpg with w and h user defined or random "
    w = int(input("Crop image's width:"))
    h = int(input("Crop image's height:"))
    r = input("Randomflag('Default:N'):")
    if r == "N" or  r == "" :
        randomflag = False
    else:
        randomflag = True
        random_num = int(input("No. of images:"))
    # w= 780;h =144; randomflag = True;random_num=2
    savedir = mkFolder(imgdir,"Crop_images")
    imgfiles, _ = getFiles(imgdir, ImgType)
    for imgfile in tqdm(imgfiles):
        print(imgfile)
        im = cv2.imread(imgfile)
        im_h,im_w,im_c = im.shape
        if im_h >= h and im_w >= w:
            if randomflag == True:
                for i in range(random_num):
                    h0 = random.randrange(0, im_h - h)
                    w0 = random.randrange(0,im_w-w)
                    crop_img = im[h0:h0 + h, w0:w0 + w]
                    filename = os.path.split(imgfile)[-1]
                    saveimg = os.path.join(savedir, filename[:-4] + '_' + str(i) + imgfile[-4:])
                    cv2.imwrite(saveimg, crop_img)
            else:
                index = 0
                for i in range(im_w//w):
                    for j in range(im_h // h):
                        h0 = 0+j*h if j*h<im_h else 0 ;w0 = 0+i*w if i*w<im_w else 0
                        crop_img = im[h0:h0+h, w0:w0+w]
                        index += 1
                        filename = os.path.split(imgfile)[-1]
                        saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index) + imgfile[-4:])
                        cv2.imwrite(saveimg, crop_img)
                # if im_w % w != 0:
                #     index += 1
                #     crop_img = im[im_h-h:im_h, im_w-w:im_w]
                #     saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index) + imgfile[-4:])
                #     cv2.imwrite(saveimg, crop_img)
                if  im_w%w != 0 or im_h%h != 0:
                    index += 1
                    crop_img = im[im_h-h:im_h, im_w-w:im_w]
                    saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index)  + imgfile[-4:])
                    cv2.imwrite(saveimg, crop_img)
        else:
            print("Image size user defined dosen't meet image shape.")

def main_img_to_video(imgdir):
    " Change images to a video "
    savedir = mkFolder(imgdir,"video")
    _,imgfiles = getFiles(imgdir,ImgType)
    imgfiles.sort()
    for id, file in enumerate(imgfiles):
        img = cv2.imread(imgdir + file)
        h, w, _ = img.shape
        if id == 0:
            path = os.path.join(savedir,'video.mp4')
            print(path)
            vid_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 1, (int(w), int(h)))
        vid_writer.write(img)
    vid_writer.release()




def main_padding_image(imgdir):
    " Padding images only for img "
    replacetxt = input("directory is replaced:")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgdir = os.path.join(root, file)
            img = cv2.imread(imgdir)
            dir = os.path.dirname(imgdir)
            size = img.shape
            h = size[0]
            w = size[1]
            WHITE = [255,0,0]
            if w > h:
                border = (w - h) // 2
                constant = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT,value = WHITE)
            else:
                border = (h - w) // 2
                constant = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=WHITE)
            imgdir = imgdir.replace(replacetxt, "Padding")

            if os.path.exists(os.path.dirname(imgdir)):
                cv2.imwrite(imgdir, constant)
            else:
                os.makedirs(dir.replace(replacetxt, "Padding"))
                cv2.imwrite(imgdir, constant)



def main_resize_image(imgdir):
    " Resize images "
    savedir = mkFolder(imgdir,"Image_resize")
    imgfull,_ = getFiles(imgdir,ImgType)
    w = int(input("Image resize width:"))
    h = int(input("Image resize height:"))
    for imgdir in tqdm(imgfull):
        img = cv2.imread(imgdir)
        filename = os.path.split(imgdir)[-1]
        (imgh,imgw,_) = img.shape
        if h ==  imgh and w == imgw:
            continue
        # print(f"\nResized image:{imgdir}\n")
        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
        new_file_path =  os.path.join(savedir,f'{filename[:-4]}' +  f'{filename[-4:]}')
        cv2.imwrite(new_file_path, img)

        xmlfile = imgdir[:-4]+'.xml'
        if os.path.exists(xmlfile):
            bboxlist,ow,oh = getObjectxml(xmlfile,classes='all')
            dbboxlist = []
            for data in bboxlist:
                data = scaleBoundingBox(data, (ow,oh), (w,h))
                dbboxlist.append(data)
            xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": dbboxlist}
            createObjxml(xmldic, new_file_path[:-4]+'.xml', cls=[])


def main_rename_file_based_on_objection(filedir):
    ''' Reanme file with objects' name'''
    savedir = mkFolder(filedir,"RenFile")
    imgfilefull,_ = getFiles(filedir,ImgType)
    for imgfile in tqdm(imgfilefull):
        xmlfile = imgfile[:-4] + '.xml'
        objlist,_,_ = getObjectxml(xmlfile,classes='all')
        objnamelist = []
        for obj in objlist:
            if obj[0] not in objnamelist:
                objnamelist.append(obj[0])
        xmlfile = Path(xmlfile)
        imgfile = Path(imgfile)
        if "__o" not in  xmlfile.name:
            newxmlfilename = os.path.join( savedir,  '_'.join(objnamelist) + '__o' + xmlfile.name)
            newimgfilename = os.path.join( savedir,  '_'.join(objnamelist) + '__o' + imgfile.name)
        else:
            start = str(xmlfile.name).find('__o')
            xmlfilenewname = xmlfile.name[start:]
            imgfilenewname = imgfile.name[start:]
            newxmlfilename = os.path.join( savedir,  '_'.join(objnamelist) +  xmlfilenewname)
            newimgfilename = os.path.join( savedir,  '_'.join(objnamelist) +  imgfilenewname)
        copyfile(xmlfile,newxmlfilename )
        copyfile(imgfile, newimgfilename )

def main_split_dataset(filedir):
    ''' Devide dataset to several small datasets'''
    num = int(input(('Please input the number of Dataset will be splited:')))
    filelist,_ = getFiles(filedir,ImgType)
    num_file = len(filelist)
    condition = num_file//num
    for i in range(num):
        savedir = mkFolder(filedir, str(i))
        file_num = 0
        while file_num != condition:
            file = filelist.pop()
            move(file,savedir)
            for cpfile in findRelativeFiles(file):
                move(cpfile, savedir)
            file_num += 1
            if len(filelist) == 0:
                break


    imgfilefull, _ = getFiles(filedir, ImgType)

def main_add_figurelabel(filedir):
    ''' Add label for figure xml'''
    imgfiles, _ = getFiles(filedir, ImgType)
    label =  input("Label name:") 

    if label == "":
        label  = "temp"
        Min_x = 0
        Min_y = 0
        Max_x = 1
        Max_y = 1
    else:   
        Min_x = int(input("BOX Min_x:")) 
        Min_y = int(input("BOX Min_y:"))
        Max_x = int(input("BOX Max_x:"))
        Max_y = int(input("BOX Max_y:"))
 
    for imgfile in tqdm(imgfiles):
        xmlfile = imgfile[:-4] + '.xml'
        if os.path.exists(imgfile[:-4] + '.xml'):
            bboxlist,w,h = getObjectxml(xmlfile,classes='all')
        else:
            bboxlist,w,h = [],99999,99999
        bboxlist.append([Min_x,Min_y,Max_x,Max_y,0,label])
        xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": bboxlist}
        createObjxml(xmldic, imgfile[:-4] + '.xml', cls=[])

def main_movefilestoone(imgdir,num=0):
    """
     num = 0 move
     num > 0  rename file with dir.
     Move files into father dirctory"""
    for root, dirs, files in os.walk(imgdir):
        for file in tqdm(files):
            if num == 0:
                try:
                    move(os.path.join(root, file),imgdir)
                except:
                    print(file)
            else:

                folders = []
                file_path = root
                for i in range(num):
                    file_path, folder = os.path.split(file_path)
                    if folder != "":
                        folders.append(folder)
                    else:
                        if file_path != "":
                            folders.append(file_path)
                        break
                folders.reverse()
                folders.extend([file])

                foldesaname = "_".join(folders)
                copyfile(os.path.join(root, file),file_path+"//"+foldesaname)


def main_moveconfuse(imgdir):
    " Move error files into errorsamples folder"
    savedir = mkFolder(imgdir,'errorsamples')
    for line in open(imgdir+"errorlist.txt","r"):
        file = imgdir + line.split(' ')[0]
        for cpfile in findRelativeFiles(file):
            move(cpfile, savedir)

def main_mkdirforonedir(imgdir):
    "Move one image to folder named with image file name"
    _,imgfiles = getFiles(imgdir, ImgType)

    for id,img in enumerate(imgfiles):
        savedir = mkFolder(imgdir,str(id))
        move(os.path.join(imgdir,img), savedir)
def sparsebbox(bbox):
    label = bbox[0]
    x0 = bbox[1]
    y0 = bbox[2]
    w = bbox[3]
    h = bbox[4]
    return x0,y0,w,h

def saveCropObject(imgdir, file, savedir,xmlfile="crop.xml"):
    bbox, w, h = getObjectxml(imgdir+xmlfile, classes="all")
    img = cv2.imread(imgdir + file)
    for i,box in enumerate(bbox):
        x0,y0,w,h = sparsebbox(box)
        cv2.imwrite(str(Path(savedir,Path(file.replace(file[-4:],f"_{i}.jpg")))), img[x0:x0+w, y0:y0+h])

def saveCropObject(imgdir, file, savedir,num,wh):
    img = cv2.imread(imgdir + file)
    (h1,w1,_) = img.shape
    w,h = wh
    inter = int((w1-w) / num)


    for i in range(num):
        if i == 0:
            x0 = 0;y0 = 0;x1=w;y1 = h
        else:
            x0 = i * inter;
            y0 = 0
            x1 = i * inter+w;
            y1 = h
        cv2.imwrite(str(Path(savedir, Path(file.replace(file[-4:], f"_{i}.jpg")))), img[ y0:y1,x0:x1])

def main_cropfixedroi(imgdir,cls=["other"]):
    '''Move crop.xml file into the target dir.
       Crop imgs with xml，
       Note：Label Name: other
    '''
    num = int(input("Please input splited figures number"))

    savedir = mkFolder(imgdir,cls[0])
    _,imgfiles = getFiles(imgdir,ImgType)
    for file in tqdm(imgfiles):
        try:
            saveCropObject(imgdir, file,savedir,num,wh=[144,144])
        except Exception as e:
            print(e)
            print(traceback.format_exc())

def main_cropsquareroi(imgdir,clsname=["temp"]):
    "Move crop.xml file into the target dir."
    _,imgfiles = getFiles(imgdir,ImgType)
    for file in tqdm(imgfiles):
        try:
            for cls in clsname:
                savesquareImg(imgdir, file, cls,scale=0,square=False,fixroi =True)
        except Exception as e:
            print(e)
            print(traceback.format_exc())


def main_crop_square_image(imgdir,keepnoobj):
    '''
        Crop square imge, keep object in xml
        output : square img xml
    '''
    savedir = mkFolder(imgdir, "square_dataset")
    imgfull, imgfiles = getFiles(imgdir, ImgType)
    total = len(imgfiles)
    cls = input("Class you want to save(person,mask.Note: has the same sort as yaml): ")
    cls = cls.split(",")
    for id, file in enumerate(imgfiles):
        img = cv2.imread(imgdir + file)
        print("%d/%d Currrent image: %s" % (id + 1, total, file))
        bbox, w, h = getObjectxml(imgfull[id].replace(file[-4:], ".xml"), classes="all")
        #算图的位置
        s = min(w, h);
        l = max(w, h);
        n = int(np.round(w / h + 0.5)) if w > h else int(np.round(h / w + 0.5));
        start = np.linspace(0, l - s, n)

        for i in start:
            i = int(i)
            imgpath = str(savedir / str(file.replace(file[-4:], "_" + str(i) + file[-4:])))
            xmlpath = imgpath.replace(imgpath[-4:], ".xml")

            if w > h:
                # print(s*i-offset*(i+1),s*i-offset*(i+1)+s)
                cv2.imwrite(imgpath, img[0:s, i:i + s])
                window_xml(xmlpath, bbox, [i, 0, i + s, s], cls)
            else:
                cv2.imwrite(imgpath, img[i:i + s, 0:s], cls)
                window_xml(xmlpath, bbox, [0, i, s, i + s])

def main_movediffimg(imgdir):
    "Move different iamges by commparing files between img dirs"
    imgdir1 = input("Dir1:")+"\\"
    imgdir2 = input("Dir2:")+"\\"
    _, imgfiles1 = getFiles(imgdir1, ImgType)
    _, imgfiles2 = getFiles(imgdir2, ImgType)
    diff_files = mkFolder(imgdir1,"diff")
    for file in imgfiles2:
        if file  in imgfiles1:
            print (file)
            if os.path.exists(imgdir2+file):
                copyfile(os.path.join(imgdir2,file),os.path.join(diff_files,file))
            else:
                print(f'{file} is not found in dir 2！')
def main_mvimg2carmerafold(imgdir):
    "Move  images by to camera fold"
    img, imgname = getFiles(imgdir, ImgType)
    for i,name in enumerate(imgname):
        camera_num = name.split('_')[2]
        des = mkFolder(imgdir,str(camera_num))
        move(img[i],des)


def main_stretchfigure(imgdir):
    ""
    import numpy as np
    img, imgname = getFiles(imgdir, ImgType)
    des = mkFolder(imgdir, "res")
    w = 8192;h = 10000
    for i,name in enumerate(img):
        image = cv2.imread(name)
        h0,w0,_ = image.shape
        # 计算图像的均方差
        mean, std_dev = cv2.meanStdDev(image)
        # 提取标准差值

        gray_image = np.zeros(( h,w, 3), dtype=np.uint8)
        # 为每个通道分别赋值
        for channel in range(3):
            gray_image[:, :, channel] = int(mean[channel,0])  # 将当前通道的所有像素值设为0
        noise_image = np.random.normal(loc=0, scale=std_dev[0,0], size=(h,w,3))
        dest_image = gray_image + noise_image
        dest_image = cv2.GaussianBlur(dest_image,(17,17),11)
        x0 = random.randint(0,w-image.shape[0]);y0 = random.randint(0,h-image.shape[1])

        dest_image[y0:y0+h0, x0:x0+w0] = image

        # 提取标准差值
        cv2.imwrite(os.path.join(des,"res_"+str(i)+'.jpg'),dest_image)

def main_removeborder(dir,imsize=(1936,148)):
    '''
    Remove border and resize to user-defined size
    Args:
        files: image files:list
        imsize: resize size:tuple

    Returns:

    '''
    savedir = mkFolder(dir,"new")
    files,names = getFiles(dir,ImgType)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0,
                  cv2.IMWRITE_PNG_BILEVEL, 0,
                  cv2.IMWRITE_PNG_STRATEGY, 0,
                  cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 0]
    for file in tqdm(files):
        name = os.path.basename(file)
        im = cv2.imread(file)
        h, w, _ = im.shape
        row_indices = []
        col_indices = []
        for i in range(w):
            if calc_mode(im[:, i, 1])[1] / h > 0.9:
                col_indices.append(i)
        col_mask = np.ones((w), dtype=bool)
        col_mask[col_indices] = False
        im = im[:, col_mask]
        for i in range(h):
            if calc_mode(im[i, :, 1])[1] / w > 0.6:
                row_indices.append(i)
        row_mask = np.ones((h,), dtype=bool)
        row_mask[row_indices] = False
        im = im[row_mask]

        #
        im = cv2.resize(im, imsize)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # 设置PNG保存参数

        cv2.imwrite(os.path.join(savedir,name), im, png_params)

def main_rotate90_img(imgdir):
    '''
       Augmentation for Images
    '''
    imgfilespath,imgfiles= getFiles(imgdir,ImgType)
    savedir = mkFolder(imgdir, "rotate90")
    for id,imgfile in enumerate(imgfilespath):
        im = cv2.imread(imgfile)

        files = findRelativeFiles(imgfile)
        xmldir = imgdir + f"{imgfiles[id][:-4]}.xml"
        for file in files:
            if ".xml" in file:
                objectlist,w,h = getObjectxml(xmldir,classes='all')
        img, label = rotate90(im,targets=objectlist)
        im_h, im_w, _ = img.shape

        xmldic = {"size": {"w": str(im_w), "h": str(im_h), "c": str(3)}, "object": label}
        imgfile = str(savedir / imgfiles[id])
        cv2.imwrite(imgfile,img)
        createObjxml(xmldic, imgfile[:-4]+".xml", xmlfile=None)
    return 0

def main_samenamefile(dir):
    '''
        Move samme file name into one folder!
    '''
    format = input("Mov same name files into one folder(e.g. *.tif):")
    typelist = [i for i in format.split(",")]
    imgdirs, _ = getFiles(dir, typelist)
    samedir = mkFolder(dir,"same")

    for img in tqdm(imgdirs):
        files = findRelativeFiles(img)
        if len(files) > 1:
            # move(img, samedir)
            for file in files:
                move(file,samedir)


def main_compareimgdiff(dir):
    '''
        Move samme file name into one folder!
    '''
    format = input("Mov same name files into one folder(e.g. *.tif):")
    typelist = [i for i in format.split(",")]
    imgdirs, _ = getFiles(dir, typelist)
    res = []

    for img in tqdm(imgdirs):
        files = findRelativeFiles(img)
        if len(files) == 2:
            image1 = cv2.imread(files[0])
            image2 = cv2.imread(files[1])
            image1  = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2  = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            subtracted_image = cv2.absdiff(image1, image2)
            image1 = zscore(image1)
            image2 = zscore(image2)
            plt.imsave(files[0].replace(files[0][-4:],"_.jpg"),subtracted_image, cmap='gray')
            # 计算两幅图像的灰度直方图
            hist1,vector1 = calculate_histogram(image1)
            hist2,vector2 = calculate_histogram(image2)

            # 计算灰度直方图相似度
            similarity_hist,similarity_cosine = calculate_histogram_similarity(hist1, hist2,vector1,vector2)
            res.append([files[0],similarity_hist,similarity_cosine,np.max(subtracted_image)])
    res = pd.DataFrame(res)
    res.to_csv(dir+"res.csv")

def main_Huaatjsontoxml(datadir):
    typelist = ["*.json"]
    for root, dirs, files in os.walk(datadir):
        if dirs ==[]:
            jsondirs,_ = getFiles(root,typelist)
            if jsondirs == []:
                print(f"No json files: {root}")
            else:
                for jsonfile in jsondirs:
                    with open(jsonfile, 'r') as defectdic:
                        data = json.load(defectdic)
                    new_bbox = []
                    for defect in data["anno"]:
                        box = eval(defect["coor"])
                        defect_label = defect["label"][3:].lower()
                        new_xmin, new_ymin, new_xmax, new_ymax = box[0][0],box[0][1],box[1][0],box[1][1]
                        temp = [ new_xmin, new_ymin, new_xmax, new_ymax,1, defect_label]
                        new_bbox.append(temp)
                    print(jsonfile[:-8] + "jpg")
                    img = cv2.imread(jsonfile[:-8] + "jpg")
                    (h, w, _) = img.shape
                    xmldic = {"size": {"w": str(w), "h": str(h), "c": str(3)}, "object": new_bbox}
                    createObjxml(xmldic, jsonfile[:-4]+".xml", [])
    # print(data)
    return


def find_keys_by_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys

def main_S3CSV2XML(folddir):
    '''
    读入数据，解析成xml文件
    '''

    defect_dic = {
        "dqp": "HA",
        "qp": "HA",
        "bbqp": "HB",
        "jqtw": "HH",
        "clxtp": "IA",
        "zgxtp": "IB",
        "wdxtp": "IC",
        "bdxtp": "ID",
        "tph": "IF",
        "htp": "IG",
        "hb": "IK",
        "zh": "JJ",
        "hs": "KA",
        "cs": "KB",
        "zp": "KE",
        "dr": "KH",
        "kd": "KI",
        "bs": "SA",
        "bx": "SF",
        "jz": "HD",
        "dtdw": "E2",
        "dzyr": "LA",
        "zdxtp": "II",
        "yh": "JA",
        "nt": "IE",
        "zd": "NA",
        "hx": "HD",
        "bg": "WW"
    }

    csvfiles, _ = getFiles(folddir, ["*.csv"])
    for csvf in csvfiles:
        if os.path.exists(csvf[:-4]):
            print(csvf)
            db = pd.read_csv(csvf, encoding='gbk')
            db1 = db.iloc[1:]
            # db1= db1.loc[:, ['out_mat_no':'defectname','topx0':]]
            print(db1.keys())
            for i, d in db1.iterrows():
                jpgname = str(d["imagefile"])
                roix0 = str(d["roix0"])
                roiy0 = str(d["roiy0"])
                roix1 = str(d["roix1"])
                roiy1 = str(d["roiy1"])
                cls = str(d["defectclasscode"])
                keys = find_keys_by_value(defect_dic,  cls)
                bbox = [[roix0, roiy0, roix1, roiy1, 0, keys[0]]]
                print(bbox)
                xmldic = {"size": {"w": str(1536), "h": str(148), "c": str(3)}, "object": bbox}
                createObjxml(xmldic,  csvf[:-4] + "/" + jpgname, cls=[], xmlfile=None)
            # move(csvf, csvf[:-4])
        else:
            print("Error: no folder found! ")
            
def adjustw_ratio(x1,y1,x2,y2,ratio,imgw):
    w = x2-x1;h = y2-y1
    print(w/h)
    delta_w = int((h*ratio - w)/2)
    if x1 - delta_w < 0 :
        x2 = x2 + delta_w*2-x1;x1 = 0
    elif x2 + delta_w > imgw :
        x2 =  imgw ;x1 = imgw-delta_w*2
    else :
        x2 = x2+delta_w;x1 = x1-delta_w 
    w = x2-x1;h = y2-y1
    print(ratio,w/h)
        
    # #TODOLIST  
        
    # x2 = x2 + delta_w

    return x1,y1,x2,y2           
            
def adjustwh(x1,y1,x2,y2,thred):
    w = x2-x1;h=y2-y1
    ratio = int(max(w,h)/min(w,h))
    print(f"ratio:{ratio}")
    if ratio > thred:
        ratio = thred
        flag = True
        if w > h:
            delta_h = int((w/ratio-h)/2+1)
            if y1 - delta_h < 0:
                y1 = 0; y2= y2 + 2*delta_h
            else:
                y1 = y1-delta_h;y2=y2+delta_h

        else:
            delta_w = int((h/ratio-h)/2+1)
            if x1 - delta_w < 0:
                x1 = 0; x2= x2 + 2*delta_w
            else:
                x1 = x1-delta_w;x2=x2+delta_w
    else:
        flag = False
    return x1,y1,x2,y2,flag


def main_adjustwh(file_dir):
    """
    Adjust w h to a proper raio for Yolo label
    """
    xml_dirs, _ = getFiles(file_dir, LabelType)
    savedir = mkFolder(file_dir, "newxml")

    for xml in xml_dirs:
        bboxes, w, h = getObjectxml(xml, "all")
        new_bboxes = []
        flag_list = []
        for bbox in bboxes:
            # bbox[0], bbox[1], bbox[2], bbox[3], flag = adjustwh(bbox[0], bbox[1], bbox[2], bbox[3], thred=200)
            bbox[0], bbox[1], bbox[2], bbox[3]= adjustw_ratio(bbox[0], bbox[1], bbox[2], bbox[3],0.15,w)
            temp = [bbox[0], bbox[1], bbox[2], bbox[3], 1, bbox[5]]
            new_bboxes.append(temp)
            xmldic = {"size": {"w": str(w), "h": str(h), "c": "3"}, "object": new_bboxes}
            createObjxml(xmldic, os.path.join(savedir, os.path.basename(xml)))


def split_list_randomly(lst, num_parts):
    random.shuffle(lst)  # 随机打乱列表顺序
    avg = len(lst) // num_parts  # 计算每份的平均长度
    remainder = len(lst) % num_parts  # 计算剩余的元素个数

    result = []
    start = 0
    for i in range(num_parts):
        length = avg + 1 if i < remainder else avg  # 根据剩余元素个数决定每份的长度
        result.append(lst[start:start + length])
        start += length
    return result


def main_splitdataset(file_dir):
    """
    Random Move files to different folders.
    """
    img_dirs, _ = getFiles(file_dir, ImgType)
    n = int(input("Input the number you want to divide:"))
    group_list = split_list_randomly(img_dirs,n)
    for i,group in enumerate(group_list):
        savedir = mkFolder(file_dir,str(i))
        for f in group:
            for fs in findRelativeFiles(f):
                move(fs,savedir)

def main_removeduplicate(xmldir):
    """
    Remove duplicate "by" and adjust whbuy

    """
    _,xmlfiles = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    remDuplicateObjectxml(xmldir,xmlfiles,cls,isSavas=True)


def remDuplicateObjectxml(xmldir, xmlfiles, classes, isSavas=True):
    '''
    Description: remove object from xmlfile in VOC
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        xmldir[str]:xml file directory
        xmlfile[],classes
    Return:
        NaN
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/'
        xmlfiles = glob.glob(filedir + '*.xml')
        remObjectxml(filedir,xmlfiles,["person"],isSavas=False)
    '''

    # xmlfile = os.path.join(xmldir, xmlfile)
    savedir = mkFolder(xmldir, "rem_copy")
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        xmlpath = xmldir + xmlfile
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        objects = root.findall("object")
        isfindflag = 0
        seen = []
        for obj in objects:
            name = obj.find('name').text
            print(name)
            if name in classes:
                if name in seen:
                    root.remove(obj)
                else:
                    seen.append(name)
                isfindflag = 1

        if isfindflag == 1:
            print(xmlpath, os.path.join(savedir, xmlfile))
            copyfile(xmlpath, os.path.join(savedir, xmlfile))
            for cpfile in findRelativeFiles(xmlfile[:-4]):
                copyfile(cpfile, savedir)
            tree.write(xmlpath)

def main_adjustobjectxml(xmldir):

    """

    Args:
        file_dir:

    Returns:

    """
    xmlfiles,_ = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    adjustbbox(xmldir,xmlfiles,cls)

def adjustbbox(imgdir,xmlfiles,cls):

    os.chdir(os.path.dirname(imgdir))

    # 使用OpenCV读取图片

    for id, xmlfile  in enumerate(tqdm(xmlfiles)):

        bbox, w, h = getObjectxml(xmlfile,classes="all")
        bbox_list = []
        for i, bbox in enumerate(bbox):
            print(bbox[5]) 
            if bbox[5] in cls:

                b = (float(bbox[0])-10, float(0),float(bbox[2])+10,  float(h),0.5,bbox[5])
                print(b)
            else:
                b = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),bbox[5])
            bbox_list.append(b)
        xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": bbox_list}

        createObjxml(xmldic, xmlfile)

def main_masknonroi(imgdir):
    '''
    Mask nonroi with other color
    '''
    imgdirs,_ = getFiles(imgdir,ImgType)
    color = (255,0,0)
    savedir = mkFolder(imgdir,"newimg")
    masknonroi(imgdirs,savedir,color)



def masknonroi(imgdirs,savedir,color=(0,0,255)):

    for imgdir in tqdm(imgdirs):
        img,bboxes,w,h = preprocess(imgdir)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = color
        for i,bbox in enumerate(bboxes):
            xmin,ymin,xmax,ymax = bbox[1],bbox[2],bbox[3],bbox[4]
            bg[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(os.path.join(savedir,str(i)+"_" + os.path.basename(imgdir)),bg)

def main_movexmlverified(xmldir):
    '''
        Move file included object to object dir  
    '''
    savedir = mkFolder(xmldir,"verify")
    xmlfiles,_ = getFiles(xmldir,LabelType)
    chkVerifiedxml(xmlfiles,savedir)

def main_checkfigurequality(xmldir):
    '''
       Check Quality
    '''
    savedir = mkFolder(xmldir,"QualityFigure")
    imgfiles,_ = getFiles(xmldir,ImgType)
    chkFigurequality(imgfiles,savedir)

def chkFigurequality(imgfiles,savedir):
    for imgfile in imgfiles:
        image_path_encoded = imgfile.encode('utf-8').decode('utf-8')
        img = cv2.imread(image_path_encoded)
        red_channel = img[:, :, 2]  # 第三个通道是红色通道
        height, width, channels = img.shape
    
        # 计算红色通道最后五个像素的起始位置
        start_x = width - 5
        # 获取红色通道的最后五个像素的数值
        last_five_red_pixels = red_channel[-1, start_x:start_x+5]
        variance = np.var(last_five_red_pixels)
        print(variance)
        if variance == 0:
            for file in findRelativeFiles(imgfile):
                move(file,savedir)
        else:
            pass

def test():
    print("test")
    return "ok"

    

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"C:\Users\chaos\Desktop\new\image\temp/"
        # pass
    try:
        if action == "getFrame": # Extract frame from 
            print(main_extract_frame_from_video.__doc__)
            main_extract_frame_from_video(file_dir)
        elif action == "main_checkfigurequality":#main_checkfigurequality
            print(main_checkfigurequality.__doc__)
            main_checkfigurequality(file_dir)
        elif action =="main_movexmlverified":
            print(main_movexmlverified.__doc__)
            main_movexmlverified(file_dir)
        elif action =="main_masknonroi":
            print(main_masknonroi.__doc__)
            main_masknonroi(file_dir)
        elif action == "adjustobjectxml":
            print(main_adjustobjectxml.__doc__)
            # main_removeduplicate(file_dir)
            main_adjustobjectxml(file_dir)
        elif action == "splitdataset":
            print(main_splitdataset.__doc__)
            main_splitdataset(file_dir)
        elif action == "main_adjustwh":
            print(main_adjustwh.__doc__)
            main_adjustwh(file_dir)
        elif action =="S32XML":#
            print(main_samenamefile.__doc__)
            main_S3CSV2XML(file_dir)
        elif action == "main_Huaatjsontoxml":#Huaatjsontoxml
            main_Huaatjsontoxml(file_dir)
        elif action == "compare2img":#
            main_compareimgdiff(file_dir)
        elif action =="samenamefile":
            print(main_samenamefile.__doc__)
            main_samenamefile(file_dir)
        elif action == "main_remove_obj_from_xml":
            print(main_remove_obj_from_xml.__doc__)
            main_remove_obj_from_xml(file_dir)
        elif action == "voc2yolo":
            print(main_change_voc_to_yolo.__doc__)
            main_change_voc_to_yolo(file_dir)
        elif action == "main_change_cls_name":
            print(main_change_cls_name.__doc__)
            main_change_cls_name(file_dir)
        elif action == "changefilename":#changefilename
            print(main_rename_file_based_on_objection.__doc__)
            main_rename_file_based_on_objection(file_dir)
        elif action == 'splitYoloTrainVal':#splitYoloTrainVal
            print(main_yolo_train_val_set.__doc__)
            main_yolo_train_val_set(file_dir,task='trainval')
        elif action == "cropObject": #cropObject
            print(main_crop_object_img.__doc__)
            main_crop_object_img(file_dir)
        elif action == "plotBBox":#plotBBox
            print(main_plot_bbox.__doc__)
            main_plot_bbox(file_dir)
        elif action == "checklabelxml":#checklabelxml
            print(main_check_label_xml.__doc__)
            main_check_label_xml(file_dir)
        elif action == "squareimg":#squareimg
            print(main_create_square_image_samples.__doc__)
            main_create_square_image_samples_one_pic(file_dir)
        elif action == "plotinferres":
            print(main_plot_infer_res.__doc__)
            main_plot_infer_res(file_dir)
        elif action == "changeHSV":
            print(main_change_hsv.__doc__)
            main_change_hsv(file_dir)
        elif action == "clipsquareimage":#clipsquareimage
            print(main_change_hsv.__doc__)
            main_crop_square_image(file_dir)
        elif action == "changeYolo2Voc":
            print(main_change_yolo_to_voc.__doc__)
            main_change_yolo_to_voc(file_dir)
        elif action == "reduceVdieoFrame":
            print(main_video2video.__doc__)
            main_video2video(file_dir)
        elif action == "movObject":#movObject
            print(main_movobject.__doc__)
            main_movobject(file_dir)
        elif action == "remUnusedXML":
            print(main_remunusedfile.__doc__)
            main_remunusedfile(file_dir)
        elif action == "imagefiter":#imagefiter
            print(main_imagesize_filter.__doc__)
            main_imagesize_filter(file_dir)
        elif action == "splitimages":#splitimages
            print(main_split_images.__doc__)
            main_split_images(file_dir)
        elif action == "imgtovideo":
            print(main_img_to_video.__doc__)
            main_img_to_video(file_dir)
        elif action == "padding_image":#
            print(main_padding_image.__doc__)
            main_padding_image(file_dir)
        elif action == "resizeimage":#resizeimage
            print(main_resize_image.__doc__)
            main_resize_image(file_dir)
        elif action == "main_split_dataset":#
            print(main_split_dataset.__doc__)
            main_split_dataset(file_dir)
        elif action == "sobel_x":#
            print(main_change_sobelx.__doc__)
            main_change_sobelx(file_dir)
        elif action == "addfigurelabel":#addfigurelabel
            print(main_add_figurelabel.__doc__)
            main_add_figurelabel(file_dir)
        elif action == "movefilestoone":  #
            print(main_movefilestoone.__doc__)
            main_movefilestoone(file_dir)
        elif action == "moveerrorfiles":  # moveerrorfiles
            print(main_moveconfuse.__doc__)
            main_moveconfuse(file_dir)
        elif action == "mkdirforonedir":  #mkdirforonedir
            print(main_mkdirforonedir.__doc__)
            main_mkdirforonedir(file_dir)
        elif action == "movdiffimg":#movdiffimg"
            print(main_movediffimg.__doc__)
            main_movediffimg(file_dir)
        elif action == "mvimg2carmerafold":#
            print(main_mvimg2carmerafold.__doc__)
            main_mvimg2carmerafold(file_dir)
        elif action =="varscales":#varscales
            print(main_stretchfigure.__doc__)
            main_stretchfigure(file_dir)
        elif action == "cropfixedroi":#
            print(main_cropfixedroi.__doc__)
            main_cropfixedroi(file_dir)
        elif action == "imgchangetojpg":#
            print(main_imgchangetojpg.__doc__)
            main_imgchangetojpg(file_dir)
        elif action == "removeborder":#
            print(main_removeborder.__doc__)
            main_removeborder(file_dir)
        elif action == "rotate90img":
            print(main_rotate90_img.__doc__)
            main_rotate90_img(file_dir)
        elif action == "":
            print(main_movefilestoone.__doc__)
            main_movefilestoone(file_dir)
            main_movobject(file_dir,autoflag="?",numclass=99)
        elif action == "test":
            test()
            

    except Exception as e:
        print(e)
        print(traceback.format_exc())

    os.system("pause")

import glob
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from utils_xml import *
from utils_math import *
from utils_cv import *
from shutil import copyfile,move
from sklearn.model_selection import train_test_split
import random
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

ImgType = ['*.jpg','*.jpeg','*.tif','*.png','*.bmp']
VideoType = ['*.avi','*.mp4','hav','.h5']
LabelType = ['*.xml']
pyscriptpath = os.path.split(os.path.realpath(__file__))[0]
configpath = os.path.join(pyscriptpath,"config.json")

with open(configpath, 'r') as c:
    config = json.load(c)

def preprocess(imgdir):
    xmldir = imgdir.replace(imgdir[-4:],".xml")
    objectlist,w,h = getObjectxml(xmldir,"all")
    img = cv2.imread(imgdir)
    return img,objectlist,w,h

def calc_mode(nums):
    '''
    Calculate mode
    :param nums: num:list
    :return: mode,mode_count
    '''
    unique_values, value_counts = np.unique(nums, return_counts=True)

    # 找到频次数组中的最大值索引
    max_count_index = np.argmax(value_counts)
    # 返回众数
    mode = unique_values[max_count_index]
    mode_count = value_counts[max_count_index]
    return mode,mode_count

def window_xml(xmlpath,bboxes,window,cls=["person"]):
    h = window[2]-window[0];w = window[3]-window[1];c=3

    new_bbox = []
    for bbox in bboxes:
        new_xmin = max(bbox[1]-window[0],0);new_ymin = max(bbox[2]-window[1],0)
        new_xmax = min(bbox[3]-window[0],w);new_ymax = min(bbox[4]-window[1],h)
        if new_xmin >= new_xmax or new_ymin >= new_ymax :
            pass
        else:
            if bbox[0] in cls:
                temp = [new_xmin,new_ymin,new_xmax,new_ymax,1,cls.index(bbox[0])]
                new_bbox.append(temp)
    xmldic = {"size":{"w":str(w),"h":str(h),"c":str(c)},"object":new_bbox}
    createObjxml(xmldic,xmlpath,cls)

def scaleBoundingBox(data,oimg,dimg):
    '''
    Description: Scale the bbox size from origin box width and height to destination。
    Author: Yujin Wang
    Date: 2022-10-10
    Args:
        data[list]:bbox list
        oimg[tuple]:Origin image shape
        dimg[tuple]:destination image shape
    Return:
        data[list]:bbox list
    Usage:
    '''
    ow,oh = oimg
    w,h = dimg
    w_scale = float(w / ow);
    h_scale = float(h / oh)
    x1,y1,x2,y2 = float(data[1]),float(data[2]),float(data[3]),float(data[4])
    data[1] = round(x1 * w_scale);
    data[2] = round(y1 * h_scale) ;
    data[3] = round(x2 * w_scale) ;
    data[4] = round(y2 * h_scale)
    return data

def findRelativeFiles(filepath):
    '''
    Description: Find these files which has the same name as the input file  
    Author: Yujin Wang
    Date: 2022-02-15
    Args:
        filepath[str]:"C:/a.txt"
    Return:
        relative files[list]
    Usage:
    '''
    filename,type = os.path.splitext(filepath)
    relativefiles = glob.glob(filename+'.*')
    return relativefiles


def writeFile(filedir,data):
    '''
    Description: Write samplesets failes in a txt 
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        filedir: txt file
        data: text
    Return:
        NaN
    Usage:
        train_files, val_files = train_test_split(jpgfiles, test_size=0.1, random_state=55)
        sampleset(train_files,"./",fn = 'train.txt')
    '''
    f = open(filedir,'w')
    for i in data:
        f.write(i)
        f.write("\n")
    f.close()

def mkFolder(dir,foldername):
    '''
    Description: Make a folder
    Author: Yujin Wang
    Date: 2022-02-13
    Args:
        dir[str]:folder directory
        foldername[str]:folder name
    Return:
        new folder directory
    Usage:
    '''
    try:
        savedir = Path(dir + foldername)
        savedir.mkdir(parents=True, exist_ok=True)
    except:
        savedir = Path(dir + "_")
        savedir.mkdir(parents=True, exist_ok=True)
    return savedir

def getFiles(folddir,typelist):
    '''
    Description: get files in folder
    Author: Yujin Wang
    Date: 2022-02-13
    Args:
        folddir[str]:  folder directory
    Return:
        files list:(path, name)
    Usage:
    '''
    files = []

    folddir = folddir + os.path.sep
    for type in typelist:
        files.extend(glob.glob(folddir + type))
    files_wodir = [Path(i.replace("\\", "/").split("/")[-1].split(".json")[0]) for i in files]
    return files,files_wodir


def renFile(filedir,savedir,format,label,id=0):
    '''
    Description:
        Rename file in filedir
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        filedir[str]:files directory
        format[str]: file format
        outfiledir[str]:output dir, if ='', replace original file
    Return:
        NaN
    Usage:
        renFile(filedir,'.jpg')
    '''
    _,files = getFiles(filedir, [format])
    total = len(files)
    for _,file in enumerate(files):
        print("%d/%d Currrent image: %s" %(id,total,file))
        str1 = file[:-3]+'*'
        duplicatefiles = glob.glob(filedir+str1)
        try:
            if label != '':
                newname = os.path.join(savedir , str(id)+ '_' + label)
            else:
                
                newname = os.path.join(savedir , str(id)+ '_' + file[:-4])

            for file in duplicatefiles:
                copyfile(file,newname + file[-4:])  
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        id += 1



def getFrame(dir,flielist,intertime=100,timeToStart = 1):
    '''
    Description: Extract frame from video
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        dir[str]: video dir.
        flielist[list]:video list
        savedir[str]: frame save directory
    Return:
        NaN
    Usage:
        avi_list =  glob.glob(DocDir+".avi")
        filelist = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in avi_list[0]]
        print (filelist)
        getFrame(avi_list[0],filelist,savedir)
    '''
    savedir = mkFolder(dir,"frame")
    num = 0
    for index,file in enumerate(flielist):
        num += 1
        cap = cv2.VideoCapture()
        print (file)
        cap.open(dir+file)
        if cap.isOpened() != True:
            os._exit(-1)
        
        #get the total numbers of frame
        totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print ("the number of frames is {}".format(totalFrameNumber))

        #get the frame rate
        rate = cap.get(cv2.CAP_PROP_FPS)
        print ("the frame rate is {} fps".format(rate))
        
        #set the start frame to read the video
        
        frameToStart = int(timeToStart * rate)+1
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
        except:
            print("Start time is gt than video total time.")
        
        # get each frames and save
        frame_num = 0 
        date = time.strftime("%Y-%m-%d", time.localtime())
        while True:
            
            ret, frame = cap.read()
            if ret != True:
                break
            frametime = round((frame_num +frameToStart-1)/ rate,2)
            if intertime == 0:
                filename = str(frame_num) + "_" +file[:-4] + "_" + str(frametime).replace('.','p') + ".jpg"
                img_path = os.path.join(savedir ,filename)
                # print (img_path)
                cv2.imwrite(img_path,frame)
            else:    
                if frame_num % (intertime*rate) == 0:
                    filename = file[:-4] + "_" + str(frametime).replace('.','p') + ".jpg"
                    img_path = os.path.join(savedir ,filename)
                    print (img_path)
                    cv2.imwrite(img_path,frame)
            frame_num += 1
        
            # wait 10 ms and if get 'q' from keyboard  break the circle
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()


def Yolo2VOC(imgfiles,classes):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args:
        yolofile[str]: .xml file from labelimg
        classes[list]: Class name
    Return:
        obj[list]: obeject list,[['person', 592, 657, 726, 1077],['person', 592, 657, 726, 1077]]
    Usage:
        bboxlist = getObjectxml(yolofile,classes)
    '''

    # print ("Current process file:",yolofile)
    total = len(imgfiles)
    id = 1
    for imgfile in imgfiles:
        try:
            print("%d/%d Currrent image: %s" % (id, total, imgfile))
            im = cv2.imread(imgfile)
            im_h,im_w,im_c = im.shape
            bbox = []
            id += 1

            for line in open(imgfile[:-4]+'.txt'):

                clsid,cx,cy,w,h = [float(i) for i in line.split()]
                xmin,ymin,xmax,ymax =  xywh2xyxy([im_h,im_w],[cx,cy,w,h])
                bbox.append([xmin,ymin,xmax,ymax,0,clsid])
            xmldic = {"size": {"w": str(im_w), "h": str(im_h), "c": str(im_c)}, "object": bbox}
            createObjxml(xmldic, imgfile, cls=classes, xmlfile=None)

        except Exception as e:
            print(e)
            print("Error:imgfile:",imgfile)


def VOC2Yolo(xmlfiles,classes='all'):
    '''
    Description: Change xml to yolo format
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        xmlfiles[list]: xml files
        classes[list]: save classes 
    Return:
        NaN
    Usage:
        xmlfiles = glob.glob("./annotations" + '*.xml')
        classes = ["mask","nomask"]
        VOC2Yolo(xmlfiles,classes)
    '''
    total = len(xmlfiles)
    id = 1
    for file in xmlfiles :
        file = file.replace("\\", "/")
        # a = cv2.imread(file.replace(".xml",".jpg"))
        print("%d/%d Currrent image: %s" %(id,total,file))
        out_file = open(file.replace(file[-4:],".txt"),'w') 
        bboxlist,width,height = getObjectxml(file,classes)
        for bbox in bboxlist:
            try:
                cls_id = classes.index(bbox[0])
                b = (float(bbox[1]), float(bbox[3]), float(bbox[2]), float(bbox[4]))
                bb = xyxy2xywh((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            except:
                print("No object found in xml, file:%s" %(file))
        id += 1
        out_file.close()

def sampleset_paddle(filelist,dir,fn = "train.txt"):
    '''
    Description: Write samplesets failes in a txt 
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        filelist[list]:xml file list
        dir[str]: Directory of txt file
    Return:
        NaN
    Usage:
        train_files, val_files = train_test_split(jpgfiles, test_size=0.1, random_state=55)
        sampleset_paddle(train_files,"./",fn = 'train.txt')
    '''
    savedir = dir
    try:
        os.mkdir(savedir)
    except:
        pass
    fd = dir +'/'+ fn
    f = open(fd,'w')
    for i in filelist:
        line = i + " "+i.replace(".jpg",".xml") +"\n"
        f.write(line)
    f.close()
    print(fn + "is Done")
 
def savecopy(filelist,file2dir):
    '''
    Description: Save files as files with index.
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        filelist[list]:files
        file2dir[str]: files directory
    Return:
        NaN
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'    
        xmlfiles = glob.glob(filedir + '*.xml')
        savecopy(xmlfiles,filedir+"copy/")
    '''
    savedir = file2dir
    try:
        os.mkdir(savedir)
    except:
        pass
    index = 0
    for file1 in filelist:
        file2xml = file2dir + str(index) + '.xml'
        file2jpg = file2dir + str(index) + '.jpg'
        copyfile(file1,file2xml)
        copyfile(file1.replace('.xml','.jpg'),file2jpg)
        index += 1

def getImgMaxLongEdge(imgpath):
    '''
    Description: Get image shape information.
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        imgpath[str]:image path
    Return:
        h,w,h//w,h%w,'w'[tupel]:
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'
        xmlfiles = glob.glob(filedir + '*.xml')
        savecopy(xmlfiles,filedir+"copy/")
    '''

    img = cv2.imread(imgpath)
    (h,w,_) = img.shape
    if h>w:
        return h,w,h//w,h%w,'w'
    if h<=w:
        return w,h,w//h,w%h,'v'

def createSquarImg(imgfiles,pob=1,flip = ['v','h','vh',"o"]):
    '''
    Description: Creat square image
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        imgfiles[list]:images' path list
        pob[float]:probability
        flip[list]:flip style list
    Return:
        img[img]
    Usage:

    '''
    maxedge, minedge,ratio, padding, direction = getImgMaxLongEdge(imgfiles[0])
    num = len(imgfiles[1:])
    frame_padding = (len(imgfiles[1:])-num) /2.
    if direction == "v":
        padding = np.ones([int(padding/(num)), maxedge, 3], dtype=np.uint8)
    else:
        padding = np.ones([maxedge, int(padding / (num)), 3], dtype=np.uint8) 
    padding[:,:,0] = 255
    img = cv2.imread(imgfiles[0])

    for id,img1 in enumerate(imgfiles[1:]):
        img1 = cv2.imread(img1)
        if flip[id] == "o":
            pass
        elif flip[id] == "hv":
            img1,_ = reflectimg(img1, prob=pob, fliptype='h')
            img1,_ = reflectimg(img1, prob=pob, fliptype='v')
        else:
            img1,_ = reflectimg(img1, prob=pob, fliptype=flip[id])
        # print(flip[id],img1.shape)

        if direction == "v":
            img = np.concatenate([img, padding], axis=0)
            img = np.concatenate([img, img1], axis=0)

        else:
            img = np.concatenate([img, padding], axis=1)
            img = np.concatenate([img, img1], axis=1)

    if frame_padding != 0:
        if direction == "v":
            padding = np.ones([minedge,maxedge,3], dtype=np.uint8) * 255
            img = np.concatenate([img, padding], axis=0)
            img = np.concatenate([padding,img], axis=0)
        else:
            padding = np.ones([maxedge,minedge,3], dtype=np.uint8) * 255
            img = np.concatenate([img, padding], axis=1)
            img = np.concatenate([padding, img], axis=1)

    return img

def paddingSquare(img):
    '''
    Description: add padding in image to a square image
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        img[image]
    Return:
        img[img]
    Usage:

    '''
    height, width, _ = img.shape
    if height > width:
        padding = np.ones([height, int((height - width) / 2), 3], dtype=np.uint8) * 0
        img = np.concatenate([padding, img], axis=1)
        img = np.concatenate([img, padding], axis=1)
    else:
        padding = np.ones([int((width - height) / 2), width, 3], dtype=np.uint8) * 0
        img = np.concatenate([padding, img], axis=0)
        img = np.concatenate([img, padding], axis=0)
    pad_height, pad_width, _ = img.shape
    return img,int((pad_height - height)/2),int((pad_width - width)/2)


def saveCropImgcopy(imgdir,imgfile,clsname,scale=0.1,square = True):
    '''
    Description: Crop image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
        saveimgfile: croped image is saved in dir and file name
    Return:
    Usage:
    '''

    savedir = mkFolder(imgdir,clsname +"_" +'crop')
    xmlfile = imgfile.replace(imgfile[-4:],".xml")
    objectlist,w,h = getObjectxml(imgdir + xmlfile,[clsname])
    img = cv2.imread(imgdir +imgfile)

    img,hoffset,woffset = paddingSquare(img)
    height, width, _ = img.shape

    id = 0
    if len(objectlist) > 0 and objectlist[0]!=[]:
        for object in objectlist:
            id += 1
            xmin = int(object[1])+woffset;
            ymin = int(object[2])+hoffset;
            xmax = int(object[3])+woffset;
            ymax = int(object[4])+hoffset
            h = ymax - ymin; w = xmax - xmin
            scale1 = int((max(w,h)*(scale+1)-max(w,h))/2)
            offset = int(abs((h - w) / 2))
            confidence = object[5] if object[5]!= 0 else 0
            if square == True:
                if h > w:
                    y1 = ymin - scale1 ;y2 = ymax + scale1 ;x1 = xmin - offset - scale1 ; x2 = xmax + offset + scale1
                    object = [[scale1 + offset, scale1  , w + offset + scale1 , h + scale1 , confidence,0 ]]

                else:
                    y1 = ymin - offset - scale1 ;y2 = ymax + offset + scale1 ; x1 = xmin-scale1 ;x2 = xmax+scale1
                    object = [[scale1 , scale1 +offset, w+scale1 , h+offset+scale1 , confidence, 0]]
            else:
                y1 = ymin-scale1 ; y2 = ymax+scale1 ; x1 = xmin-scale1 ; x2 = dxmax+scale1
                object = [[0,0,w,h, confidence, 0]]

            ymin = max(0, y1);ymax = min(y2, height); xmin = max(0, x1);xmax = min(x2, width)
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w, c = crop_img.shape
            xmldic = {"size": {"w": str(w), "h": str(h), "c": str(c)},
                      "object": object}
            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id)+'.jpg')
            # h,w,c = crop_img.shape

            createObjxml(xmldic,saveimg,cls=[clsname])
            cv2.imwrite(saveimg,crop_img)



def saveSquareImg(imgdir, imgfile, clsname):
    savedir = mkFolder(imgdir, clsname + "_" + 'crop')
    imgfulldir = imgdir + imgfile
    img = cv2.imread(imgfulldir)
    height, width, _ = img.shape
    edgesize = min(height,width)
    xmlfile = imgfulldir.replace(imgfulldir[-4:], ".xml")
    objectlist, w, h = getObjectxml(xmlfile, [clsname])
    id = 0
    if len(objectlist) > 0 and objectlist[0] != []:
        for objectbox in objectlist:
            id += 1
            xmin = int(objectbox[1]) ;
            ymin = int(objectbox[2]) ;
            xmax = int(objectbox[3]) ;
            ymax = int(objectbox[4]) ;
            h = ymax - ymin;
            w = xmax - xmin
            if xmin == 0 :
                xmax = edgesize
            if xmax == width:
                xmin = width - edgesize;

            if ymin == 0:
                ymax = edgesize
            if ymax == height:
                ymin = height-edgesize ;
            crop_img = img[ymin:ymax, xmin:xmax]
            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id) + '.jpg')
            cv2.imwrite(saveimg, crop_img)




def saveCropImg(imgdir, imgfile, clsname, scale=0.1, square=True,resize_img =0,fixroi = False):
    '''
    Description: Crop image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args:
        img[image]: image
        object: dest object
        saveimgfile: croped image is saved in dir and file name
    Return:
    Usage:
    '''
    def box_in_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """
        判断box (x1, y1, x2, y2)是否在box (xmin, ymin, xmax, ymax) 之中
        """
        xc = (x1+x2)/2;yc = (y1+y2)/2
        return (xmin <= xc <= xmax) and (ymin <= yc <= ymax),(xmin <= x1 <= xmax) and (xmin <= x2 <= xmax) and (ymin <= y1 <= ymax) and (ymin <= y2 <= ymax)
    savedir = mkFolder(imgdir, clsname + "_" + 'crop')
    imgfulldir = imgdir + imgfile
    img = cv2.imread(imgfulldir)
    height, width, _ = img.shape
    if fixroi == False:
        xmlfile = imgfulldir.replace(imgfulldir[-4:], ".xml")
        objectlist, w, h = getObjectxml(xmlfile, [clsname])
    else:
        try:
            xmlfile = imgdir + "crop.xml"
            objectlist, w, h = getObjectxml(xmlfile, 'all')
        except Exception as e:
            print(e)
            print(traceback.format_exc())


    img, hoffset, woffset = paddingSquare(img)
    id = 0
    if len(objectlist) > 0 and objectlist[0] != []:
        for objectbox in objectlist:
            id += 1
            xmin = int(objectbox[1]) + woffset;
            ymin = int(objectbox[2]) + hoffset;
            xmax = int(objectbox[3]) + woffset;
            ymax = int(objectbox[4]) + hoffset
            h = ymax - ymin;
            w = xmax - xmin
            scale1 = int((max(w, h) * (scale + 1) - max(w, h)) / 2)
            offset = int(abs((h - w) / 2))
            confidence = objectbox[5] if objectbox[5] != 0 else 0
            if square == True:
                if h > w:
                    y1 = ymin - scale1;
                    y2 = ymax + scale1;
                    x1 = xmin - offset - scale1;
                    x2 = xmax + offset + scale1
                    object = [[scale1 + offset, scale1, w + offset + scale1, h + scale1, confidence, 0]]

                else:
                    y1 = ymin - offset - scale1;
                    y2 = ymax + offset + scale1;
                    x1 = xmin - scale1;
                    x2 = xmax + scale1
                    object = [[scale1, scale1 + offset, w + scale1, h + offset + scale1, confidence, 0]]
            else:
                y1 = ymin - scale1;
                y2 = ymax + scale1;
                x1 = xmin - scale1;
                x2 = xmax + scale1
                object = [[0, 0, w, h, confidence, 0]]

            ymin = max(0, y1);
            ymax = min(y2, height);
            xmin = max(0, x1);
            xmax = min(x2, width)
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w, c = crop_img.shape

            classes = [clsname]
            innerobjectlist, _, _ = getObjectxml(xmlfile, "all")
            for innerbox in innerobjectlist:
                confidence = innerbox[5] if innerbox[5] != 0 else 0
                centerinflag,_ = box_in_box(innerbox[1], innerbox[2], innerbox[3], innerbox[4], objectbox[1], objectbox[2], objectbox[3],objectbox[4])
                if innerbox[0] != clsname and centerinflag:
                    if innerbox[0] not in classes:
                        classes.append(innerbox[0])
                    x1 = object[0][0] + innerbox[1]-objectbox[1]; y1 = object[0][1] + innerbox[2]- objectbox[2] ;
                    object.append([x1, y1, x1 + innerbox[3] - innerbox[1],y1 + innerbox[4] - innerbox[2], confidence ,classes.index(innerbox[0])])


            if resize_img !=0 :
                crop_img = cv2.resize(crop_img,(resize_img ,resize_img ))
            else:
                xmldic = {"size": {"w": str(w), "h": str(h), "c": str(c)},"object": object}
                # createObjxml(xmldic, saveimg, cls=classes)

            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id) + '.jpg')
            cv2.imwrite(saveimg, crop_img)




def plotRectBox(img,objectlist,names):
    '''
    Description: Plot bndbox and label in image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
    Return:
        img
    Usage:
    '''
    sys.path.append(config["yolov5"])
    from utils.plots import Annotator, colors
    annotator = Annotator(img, line_width=1, example="")
    
    for object in objectlist:
        if len(object)==6:
            label, xmin, ymin, xmax, ymax,conf =object[0],object[1], object[2], object[3], object[4],object[5]
        else:
            label, xmin, ymin, xmax, ymax,conf =object[0],object[1], object[2], object[3], object[4],0
        c = names.index(label)
        label =  f'{names[c]} ' #{conf:.2f}
        annotator.box_label([xmin, ymin, xmax, ymax], label, color=colors(c, True))
        # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        # cv2.putText(img, object[0], (int((xmin+xmax)/2),int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    return annotator.result()
    # cv_show('img',img)

def changeHSV1(img,adjh=1.0,adjs=1.0,adjv=1.1):
    # print(adjh,adjs,adjv)
    adjh = adjh+0.5;adjs =adjs+0.8;adjv = adjv+1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    
    x = np.arange(0, 256)
    
    lut_hue = ((x * adjh) % 180).astype(dtype)
    lut_sat = np.clip(x * adjs, 0, 255).astype(dtype)
    lut_val = np.clip(x * adjv, 0, 255).astype(dtype)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(im_hsv,cv2.COLOR_HSV2BGR)

def changeHSV(img):
    '''
    Description:
        Change image light
    Author: Yujin Wang
    Date: 2022-02-22
    Args:
        img[cv.img]:Gray
    Return:

    Usage:
    '''
    # print(adjh,adjs,adjv)
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    # h_norm = (hue/np.max(hue)*180).astype(dtype)
    # s_norm = (sat/np.max(sat)*255).astype(dtype)
    v_norm = (val/np.max(val)*255).astype(dtype)
    img_hsv =  cv2.merge([hue,sat,v_norm])
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


def plotFigArray(imglist:list,imgshape=(0,0)):
    '''
    Description: 
          Plot multi imgs in a figure
    Author: Yujin Wang
    Date: 2022-02-21
    Args:
        filerir[str]:file directory
    Return:
        figures
    Usage:
    '''
    height, width, _ = imglist[0].shape
    num = len(imglist)
    row,col = calcImgArray(width,height,num)
    print(row,col)
    canvas = np.ones([row*height, col*width,3], dtype=np.uint8) * 255
    total = 0
    for i in range(row):
        for j in range(col):
            if total < num:
                canvas[height*i:height*(i+1),width*j:width*(j+1)] = imglist[total]
                total += 1
            else:
                break
    
    if imgshape !=(0,0):
        canvas = cv2.resize(canvas,imgshape)
    # cv_show("canvas", canvas)
    return canvas

def Video2Video(videofile,savedir,interval,offset,scale):
    '''
    Description: Reduce frame numbers of video or reshape video size
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        videofile[str]:video path
        savedir[str]: new video path
        interval[int]: video interval number
        offset[int]: number of frames
        scale[float]:scale of video shape
    Return:
        NAN
    Usage:

    '''
    print(videofile)
    cap = cv2.VideoCapture()
    cap.open(videofile)
    rate = cap.get(cv2.CAP_PROP_FPS)
    
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Current video fps:{}".format(rate))
    print("Current video frame No.:{}".format(totalFrameNumber))
    if offset+interval > totalFrameNumber:
        print("offset+interval > totalFrameNumber")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    print(w,h)
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
            videoWriter.write(frame)
        else:
            pass
        n += 1
    cap.release()
    videoWriter.release()


# MAIN PROCESSE
def main_extract_frame_from_video(videodir):
    '''
        Extract frame from video
    '''
    _,filelist = getFiles(videodir,VideoType)
    # startid = int(input("Start image ID:"))
    interval = float(input("Interval time(s):"))
    OffsetTime = float(input("Offset time(s):"))
    getFrame(videodir,filelist,interval,OffsetTime)



def main_remove_obj_from_xml(xmldir,cls1 = "zdxtp,zgxtp,zh,kd,yh"): 
    '''
        Remove object from xml
    '''
    _,xmlfiles = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    if cls =='N':
        cls = cls1
    cls = cls.split(',')
    print(f"Classes : f{cls}")
    remObjectxml(xmldir,xmlfiles,cls,isSave=False)
 


def main_change_voc_to_yolo(xmldir,cls=[]):
    '''
        Change VOC to Yolo
    '''
    xmlfiles, _ = getFiles(xmldir, LabelType)
    if cls==[]:
        cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
        cls_name = cls_name.split(',')
    else:
        cls_name = cls
    if type(cls) != list:
        print('Input is not correct')
    VOC2Yolo(xmlfiles,cls_name)

def main_change_yolo_to_voc(imgdir):
    '''
        Change  Yolo to Voc
    '''
    imgfiles,_ = getFiles(imgdir,ImgType)
    cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
    cls_name = cls_name.split(',')
    Yolo2VOC(imgfiles,cls_name)

def main_change_cls_name(xmldir,oldcls1 = "Hs_DTDW,Hs_BS1,Hs_ZP,Hs_DQP1,Hs_ZDXTP1,Hs_BY,Hs_JZ,Hs_QP,Hs_DR,Hs_KD1,Hs_HTP,Hs_YH,Hs_DZYR,Hs_JQTW,Hs_HS,Hs_HX,Hs_CS,Hs_BDXTP,Hs_KD,Hs_BS,Hs_ZH,Hs_WDXTP,Hs_BBQP,Hs_BX,Hs_TPH,Hs_ZGXTP,Hs_HB", \
     newcls1 = "dtdw,bs,zp,dqp,zdxtp,by,jz,qp,dr,kd,htp,yh,dzyr,jqtw,hs,hx,cs,bdxtp,kd,bs,zh,wdxtp,bbqp,bx,tph,zgxtp,hb"):
    '''
        Change class name
    '''
    xmlfiles = glob.glob(xmldir+ '*.xml')
    oldcls = input("Old class:")
    newcls = input("New class:")
    if oldcls == "" and newcls =="":
        oldcls == oldcls1;newcls == newcls1
        chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSave=False)
    else:
        chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSave=False)


def plothist(data,title,imgfile,bins = [10, 20, 30, 40, 50, 70],datarange=(0,1),show= False):
    nt, _, _ = plt.hist(data, bins=51, rwidth=0.5, range=datarange, align='mid')
    plt.plot([np.mean(data), np.mean(data)], [0, np.max(nt)], ":", label="Mean")
    plt.plot([np.median(data), np.median(data)], [0, np.max(nt)], "--", label="Median")
    for i in bins:
        value = np.percentile(data, i)
        plt.plot([value, value], [0, np.max(nt)], "--", label=f"{i}%")
        plt.text(value, np.max(nt), f'{round(value, 2)}', fontsize=8, rotation=90)
    plt.xticks()
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(imgfile)
    if show:
        plt.show()
    plt.close()

def main_check_label_xml(xmldir):
    '''
        Check label xml and present histgram of confidence
    '''
    xmlfiles,_ = getFiles(xmldir, LabelType)
    noobjectfiles,cls = checkLabexml(xmlfiles)

    if len(noobjectfiles) != 0:
        savedir = mkFolder(xmldir,"noobject")
        for file in noobjectfiles:
            for cpfile in findRelativeFiles(file):
                move(cpfile,savedir)
    else:
        print("No unlabeled img found!")
    
    savedir = mkFolder(xmldir, "checkres")
    clsname =",".join(list(cls.keys()))
    print(clsname)
    for name in cls.keys():
        temp = np.array(cls[name]["confidence"])
        plothist(temp,name,savedir / f'{name}_confidence.jpg')
        np.savetxt(savedir / f'{name}_confidence_{len(cls[name]["confidence"])}.csv', np.array(cls[name]["confidence"]), delimiter=",")



def main_change_file_name(xmldir):
    '''
        Rename files
    '''
    # xmlfiles = glob.glob(xmldir+ '*.xml')
    format = '*' + input("Input file format('.jpg'):")
    label = input("Add string in file name:")
    id = int(input("Start number:"))
    savedir =mkFolder(xmldir,'rename_files')
    renFile(xmldir,savedir,format,label,id)

def main_yolo_train_val_set(imgdir,task = 'test'):
    '''
        Split train and val dataset
    '''
    mvfolder = input("Do you want to move figeures to train an val folder?(Y/N)")
    if mvfolder == "Y":
        trainFolder = mkFolder(imgdir,"train")
        valFolder = mkFolder(imgdir,"validation")

    if task != 'test':
        _, Imagefiles = getFiles(imgdir, ImgType)
        img_serverdir = input("Train and validation img in serverdir(data/.../):")
        # imgfiles_serve = [img_serverdir + i for i in imgfiles]
        samplerdir = mkFolder(imgdir, 'train_val')
        test_size = float(input("Input the ratio of val:"))
        
        
        train_files, val_files = train_test_split( Imagefiles, test_size=test_size, random_state=55)
        if mvfolder == "Y":
            for imgfile in train_files:
                for file in findRelativeFiles(os.path.join(imgdir,imgfile)):
                    move(file, trainFolder)
            for imgfile in val_files:
                for file in findRelativeFiles(os.path.join(imgdir,imgfile)):
                    move(file, valFolder)
        
        
        if  test_size  == "0":
            writeFile(samplerdir / 'test.txt', imgfiles)
            return
        
        print(img_serverdir) 
        train_files =   [str(img_serverdir)+ 'train/' + str(i) for i in train_files]
        val_files =   [str(img_serverdir) +'validation/' + str(i) for i in val_files]
        writeFile(samplerdir / 'train.txt', train_files)
        writeFile(samplerdir / 'val.txt',val_files)
    else:
        imgfiles, _ = getFiles(imgdir, ImgType)
        writeFile(imgdir + '/test.txt', imgfiles)
        return

def main_imagesize_filter(imgdir):
    filelist,_ = getFiles(imgdir,ImgType)
    imgsizelist = []
    remdir = mkFolder(imgdir,"rem")
    resdir = mkFolder(imgdir,"res")
    errdir = mkFolder(imgdir, "err")
    total = len(filelist)
    for id,file in enumerate(filelist):
        print(f'{id+1}/{total}:{file}')
        try:
            img = cv2.imread(file)
            w,h,_= img.shape
            print(f'Done')
            imgsizelist.append([file,w,h,w*h])
        except:
            move(file,errdir)

    imgsizelist = pd.DataFrame(imgsizelist,columns=["file","W","H","A"])
    imgsizelist.to_csv(resdir / "imgsize.csv")
    plothist(imgsizelist["A"], "Img_Area", resdir / "A_histgram.jpg",datarange=(min(imgsizelist["A"]),max(imgsizelist["A"])),show=True)
    lowerthr = int(input("Image lower threshold(thrXthr):"))
    upperthr = int(input("Image upper threshold(thrXthr):"))
    remimg = imgsizelist[imgsizelist["A"]<upperthr][imgsizelist["A"]>lowerthr]
    for img in remimg["file"]:
        for file in findRelativeFiles(img):
            move(file,remdir)


# def expandcropimg(image, rect, expand_ratio=1):
#     '''
#     按照一定比例(expand_ratio)将rect放大后进行裁剪
#     Author:Zhangzhe
#     '''
#     imgh, imgw, c = image.shape
#     xmin, ymin, xmax, ymax = [int(x) for x in rect]
#     org_rect = [xmin, ymin, xmax, ymax]
#     h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
#     w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
#     # if h_half > w_half * 4 / 3:
#     #     w_half = h_half * 0.75
#     center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
#     ymin = max(0, int(center[0] - h_half))
#     ymax = min(imgh - 1, int(center[0] + h_half))
#     xmin = max(0, int(center[1] - w_half))
#     xmax = min(imgw - 1, int(center[1] + w_half))
#     return image[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect

def main_crop_object_img(imgdir):
    '''
        Crop objet image(include inner objects in box)
    '''
    clsname = input("Input class name:")
    try:
        scale =  float(input("Input expand ratio (max(h,w),def.1):"))
    except:
        print("Scale def. 1")
        scale =1.0
    square = True if input("Crop image with padding(Y/N),def.Y:") == "Y" else False
    try:
         resize_img = int(input("Resize(0:no):"))
    except:
        print("Scale def. 0")
        resize_img = 0.0

    clsname = clsname.split(',')
    _,imgfiles = getFiles(imgdir,ImgType)


    for file in tqdm(imgfiles):
        try:
            for cls in clsname:
                saveCropImg(imgdir,file,cls,scale,square,resize_img)
        except Exception as e:
            print(e)
            print(traceback.format_exc())



def main_plot_bbox(imgdir):
    '''
        Plot bbox in img
    '''
    savedir = mkFolder(imgdir,"plotbbox")
    _,imgfiles = getFiles(imgdir,ImgType)
    cls = input("Class you want to plot(e.g. person,mask): ")
    cls = cls.split(",")
    total = len(imgfiles)

    # imgfiles.sort(key=lambda x: int(x.split('_')[0]))

    # 更改工作目录到图片所在目录
    os.chdir(os.path.dirname(imgdir))

    # 使用OpenCV读取图片

    for id,file in enumerate(tqdm(imgfiles)):
        print(file)
        xmlfile = imgdir + file.replace(file[-4:],".xml")
        bbox,_,_ = getObjectxml(xmlfile,cls)

        img = cv2.imread(file)
        h,w,  _ = img.shape
        print(f'w:{w} h:{h}')
        img = plotRectBox(img,bbox,names=cls)
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        imgpath = "plotbbox/" + file[:-4]+".jpg"
        print(imgpath)
        # img = plot_line(img,ptStart = (1160, 110),ptEnd = (0, 630))
        # img = plot_line(img, ptStart=(960, 35), ptEnd=(0, 339))
        cv2.imwrite(str(imgpath),img)
        if id == 0:
            path = os.path.join(savedir,'video.mp4')
            print(path)
            vid_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'acv1'), 25, (int(w), int(h)))
        vid_writer.write(img)
    vid_writer.release()
    return

def plot_line(img,ptStart = (60, 60),ptEnd = (260, 260),point_color = (0,255,255)):
    thickness = 3
    lineType = 4
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    return img


def main_create_square_image_samples_one_pic(filedir1):
    '''
        Create a square image with padding
    '''
    savedir = mkFolder(filedir1,'new_dataset')
    imgfiles1,_ = getFiles(filedir1,ImgType)
    total = len(imgfiles1)
    for id,file in enumerate(imgfiles1):

        print("%d/%d Current process file: %s" %(id+1,total,file))
        imgfilescopy = imgfiles1.copy()
        imgfilescopy.remove(file)
        imgfilepath = os.path.join(savedir, os.path.split(file)[-1][:-4] + '_square' + os.path.split(file)[-1][-4:])
        edge,minedge,fignum,padding,direction = getImgMaxLongEdge(file)
        concimgs = [file]
        concimgs.extend([file for i in range(fignum-1)])
        flip = ['v', 'h', 'vh', "o"]
        img = createSquarImg(concimgs,flip = flip)
        cv2.imwrite(imgfilepath,img)
        try:
            xmlfile = [file.replace(file[-4:],".xml") for file in concimgs]
            xmlfile = combineXMLinDirection(xmlfile,edge,fignum,padding,direction,flip = flip)
            xmlfile.write(imgfilepath.replace(file[-4:],'.xml'))
        except:
            print("No xml file is found!")



def main_create_square_image_samples(filedir1):
    '''
        Create a square image with padding
    '''
    savedir = mkFolder(filedir1,'new_dataset')
    imgfiles1,_ = getFiles(filedir1,ImgType)
    filedir2 = input("Please input another dataset(for sample balance):")
    id = int(input("Please input start id:"))
    imgfiles2,_ = getFiles(filedir2,ImgType)
    total = len(imgfiles1)
    for file in imgfiles1:

        print("%d/%d Current process file: %s" %(id,total,file))
        imgfilescopy = imgfiles1.copy()
        imgfilescopy.remove(file)
        edge,fignum,padding,direction = getImgMaxLongEdge(file)
        concimgs = [file]
        if imgfiles2:
            concimgs.extend([random.sample(imgfiles2,1)[0] for i in range(fignum-1)])
        else:
            imgfiles2 =  imgfiles1.copy()
            concimgs.extend([random.sample(imgfiles2,1)[0] for i in range(fignum-1)])
        img = createSquarImg(concimgs)
        xmlfile = [file.replace(file[-4:],"xml") for file in concimgs]
        xmlfile = combineXMLinDirection(xmlfile,edge,fignum,padding,direction)
        imgfilepath = savedir,str(id) / '.tif'
        cv2.imwrite(imgfilepath,img)
        xmlfile.write(imgfilepath.replace(file[-4:],'.xml'))
        id += 1

def main_plot_infer_res(filedir): 
    '''
        Plot infer results from multi figures
    '''
    savedir = mkFolder(filedir,'infer_res_compare')
    _,filenamelist = getFiles(filedir,ImgType)
    compareflag = True
    resdirlist = []
    while compareflag:
        resdir = input("Inference another res dir('Enter over'):")
        if resdir == "":
            break
        resdirlist.append(resdir)
        
    total = len(filenamelist)

    for id,file in enumerate(filenamelist):
        imglist = [cv2.imread(os.path.join(filedir,file))]
        for resdir in resdirlist:
            filepath = os.path.join(resdir,file)
            if os.path.exists(filepath):

                imglist.append(cv2.imread(filepath))
            else:
                print("No result found! Image path: %s" %(filepath))
        
        img = plotFigArray(imglist)
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        cv2.imwrite(os.path.join(str(savedir),"res_"+file),img)

def main_change_hsv(filedir):
    '''
        Change the light
    '''
    savedir = mkFolder(filedir,'gray')
    _,filenamelist = getFiles(filedir,ImgType)
    total = len(filenamelist)
    for id,file in enumerate(filenamelist):
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        img = cv2.imread(os.path.join(filedir,file))
        img = changeHSV(img)
        cv2.imwrite(os.path.join(str(savedir),file),img)

def sobelx(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转为灰度图
    dst = np.zeros_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv_show('ori', img)
    # cv_show('res', dst)
    img_sobel = cv2.Sobel(dst,cv2.CV_8U,1,0)
    # res = cv2.cvtColor(img_sobel,cv2.COLOR_GRAY2RGB)
    # cv_show('res',img_sobel)
    return img_sobel

def main_change_sobelx(filedir):
    '''
        Change the light
    '''
    savedir = mkFolder(filedir,'gray')
    _,filenamelist = getFiles(filedir,ImgType)
    total = len(filenamelist)
    for id,file in enumerate(filenamelist):
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        img = cv2.imread(os.path.join(filedir,file))
        img = sobelx(img)
        cv2.imwrite(os.path.join(str(savedir),file),img)


def main_video2video(videodir):
    '''
        pic to video 
    '''
    _,filelist = getFiles(videodir,VideoType)
    interval = int(input("Input interval frame number:"))
    offset = int(input("Input offset frame number:"))
    scale = float(input("Input scale ratio:"))
    savedir = mkFolder(videodir,'video')
    for file in filelist:
        Video2Video(os.path.join(videodir,file),os.path.join(savedir,file),interval,offset,scale)

def main_movobject(xmldir,autoflag="", numclass=""):
    '''
        Move file included object to object dir  
    '''
    xmlfiles,_ = getFiles(xmldir,LabelType)
    if autoflag == "":
        cls = input("Class name(label list,e.g. [person,mask]):")
    else:
        cls = autoflag
    cls = cls.split(',')

    if numclass == "":
        try:
            numclass = int(input("Number threshold of labels(def. 99):"))
        except:
            print('Default number 99 will be used!')
            numclass = 99
    for i in cls:
        xmlfiles,_ = getFiles(xmldir,LabelType)
        savedir = mkFolder(xmldir,i)
        movObjectxml(xmlfiles,i,savedir,numclass)

def main_remunusedfile(xmldir):
    '''
        Remove unused files
        label：lab；image：img
    '''
    filetype = input("Image or label will be removed:")
    if filetype == "lab":
        files,_ = getFiles(xmldir,LabelType)
    elif filetype == "img":  
        files,_ = getFiles(xmldir,ImgType)
    unuseddir = mkFolder(xmldir,'unused')
    for file in files:
        print(file)
        if len(findRelativeFiles(file)) == 1:
            move(file,unuseddir)


def main_imgchangetojpg(imgdir):
    " Change images' format to jpg "
    # gencomplexfigure(imgdir)
    # gencircle(imgdir)
    imgsdir,_ = getFiles(imgdir,ImgType)
    # quality_index = input("Please input quarlity index(0~100):")
    filetype = input("You want to change to File type('.tif'):")
    ct = []
    imgsize = []
    for img in tqdm(imgsdir):
        im = cv2.imread(img)
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        imgname = img.split('.')[0] + filetype
        t0 = time.time()
        cv2.imwrite(imgname,im,params)

        ct.append(time.time()-t0)
        imgsize.append([imgname,os.path.getsize(imgname)])
        print(f"{imgname},{os.path.getsize(imgname)}")
    ct = np.array(ct);imgsize = np.array(imgsize);
    ct_m,ct_std = np.mean(ct)*1000, np.std(ct)*1000
    print("%5.3f\t%5.3f" %(ct_m,ct_std))
    return ct_m,ct_std,imgsize

def gencomplexfigure(dir):
    for j in range(8):
        h = np.power(2,(j+1))
        n =int( 320/ h)
        block0 = np.zeros(( h,h, 3), dtype=np.uint8)
        block1 = np.zeros(( h,h, 3), dtype=np.uint8)+255
        img0 = np.hstack((block0, block1))
        img1 = np.hstack((block1, block0))
        img2 = np.vstack((img0, img1))
        row = img2
        for i in range(n-1):
            row = np.hstack((row,img2))
        img = row
        for i in range(n-1):
            img = np.vstack((img, row))
        params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        cv2.imwrite(f"{dir}{j}.bmp", img)


def gencircle(dir):
    '''

    Args:
        dir: Save dir：string

    Returns:

    '''

    height, width = 640, 640
    background = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    for j in range(8):
        radius = np.power(2, (j + 1))
        color = (255, 255, 255)  # 白色
        thickness = -1  # 填充圆形的内部
        cv2.circle(background, (center_x, center_y), radius, color, thickness)
        cv2.imwrite(f"{dir}{j}.bmp", background)

def main_split_images(imgdir):
    " Split image to several jpg with w and h user defined or random "
    w = int(input("Crop image's width:"))
    h = int(input("Crop image's height:"))
    r = input("Randomflag('Default:N'):")
    if r == "N" or  r == "" :
        randomflag = False
    else:
        randomflag = True
        random_num = int(input("No. of images:"))
    # w= 780;h =144; randomflag = True;random_num=2
    savedir = mkFolder(imgdir,"Crop_images")
    imgfiles, _ = getFiles(imgdir, ImgType)
    for imgfile in tqdm(imgfiles):
        print(imgfile)
        im = cv2.imread(imgfile)
        im_h,im_w,im_c = im.shape
        if im_h >= h and im_w >= w:
            if randomflag == True:
                for i in range(random_num):
                    h0 = random.randrange(0, im_h - h)
                    w0 = random.randrange(0,im_w-w)
                    crop_img = im[h0:h0 + h, w0:w0 + w]
                    filename = os.path.split(imgfile)[-1]
                    saveimg = os.path.join(savedir, filename[:-4] + '_' + str(i) + imgfile[-4:])
                    cv2.imwrite(saveimg, crop_img)
            else:
                index = 0
                for i in range(im_w//w):
                    for j in range(im_h // h):
                        h0 = 0+j*h if j*h<im_h else 0 ;w0 = 0+i*w if i*w<im_w else 0
                        crop_img = im[h0:h0+h, w0:w0+w]
                        index += 1
                        filename = os.path.split(imgfile)[-1]
                        saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index) + imgfile[-4:])
                        cv2.imwrite(saveimg, crop_img)
                # if im_w % w != 0:
                #     index += 1
                #     crop_img = im[im_h-h:im_h, im_w-w:im_w]
                #     saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index) + imgfile[-4:])
                #     cv2.imwrite(saveimg, crop_img)
                if  im_w%w != 0 or im_h%h != 0:
                    index += 1
                    crop_img = im[im_h-h:im_h, im_w-w:im_w]
                    saveimg = os.path.join(savedir, filename[:-4] + '_' + str(index)  + imgfile[-4:])
                    cv2.imwrite(saveimg, crop_img)
        else:
            print("Image size user defined dosen't meet image shape.")

def main_img_to_video(imgdir):
    " Change images to a video "
    savedir = mkFolder(imgdir,"video")
    _,imgfiles = getFiles(imgdir,ImgType)
    imgfiles.sort()
    for id, file in enumerate(imgfiles):
        img = cv2.imread(imgdir + file)
        h, w, _ = img.shape
        if id == 0:
            path = os.path.join(savedir,'video.mp4')
            print(path)
            vid_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 1, (int(w), int(h)))
        vid_writer.write(img)
    vid_writer.release()




def main_padding_image(imgdir):
    " Padding images only for img "
    replacetxt = input("directory is replaced:")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgdir = os.path.join(root, file)
            img = cv2.imread(imgdir)
            dir = os.path.dirname(imgdir)
            size = img.shape
            h = size[0]
            w = size[1]
            WHITE = [255,0,0]
            if w > h:
                border = (w - h) // 2
                constant = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT,value = WHITE)
            else:
                border = (h - w) // 2
                constant = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=WHITE)
            imgdir = imgdir.replace(replacetxt, "Padding")

            if os.path.exists(os.path.dirname(imgdir)):
                cv2.imwrite(imgdir, constant)
            else:
                os.makedirs(dir.replace(replacetxt, "Padding"))
                cv2.imwrite(imgdir, constant)



def main_resize_image(imgdir):
    " Resize images "
    savedir = mkFolder(imgdir,"Image_resize")
    imgfull,_ = getFiles(imgdir,ImgType)
    w = int(input("Image resize width:"))
    h = int(input("Image resize height:"))
    for imgdir in tqdm(imgfull):
        img = cv2.imread(imgdir)
        filename = os.path.split(imgdir)[-1]
        (imgh,imgw,_) = img.shape
        if h ==  imgh and w == imgw:
            continue
        # print(f"\nResized image:{imgdir}\n")
        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
        new_file_path =  os.path.join(savedir,f'{filename[:-4]}' +  f'{filename[-4:]}')
        cv2.imwrite(new_file_path, img)

        xmlfile = imgdir[:-4]+'.xml'
        if os.path.exists(xmlfile):
            bboxlist,ow,oh = getObjectxml(xmlfile,classes='all')
            dbboxlist = []
            for data in bboxlist:
                data = scaleBoundingBox(data, (ow,oh), (w,h))
                dbboxlist.append(data)
            xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": dbboxlist}
            createObjxml(xmldic, new_file_path[:-4]+'.xml', cls=[])


def main_rename_file_based_on_objection(filedir):
    ''' Reanme file with objects' name'''
    savedir = mkFolder(filedir,"RenFile")
    imgfilefull,_ = getFiles(filedir,ImgType)
    for imgfile in tqdm(imgfilefull):
        xmlfile = imgfile[:-4] + '.xml'
        objlist,_,_ = getObjectxml(xmlfile,classes='all')
        objnamelist = []
        for obj in objlist:
            if obj[0] not in objnamelist:
                objnamelist.append(obj[0])
        xmlfile = Path(xmlfile)
        imgfile = Path(imgfile)
        if "__o" not in  xmlfile.name:
            newxmlfilename = os.path.join( savedir,  '_'.join(objnamelist) + '__o' + xmlfile.name)
            newimgfilename = os.path.join( savedir,  '_'.join(objnamelist) + '__o' + imgfile.name)
        else:
            start = str(xmlfile.name).find('__o')
            xmlfilenewname = xmlfile.name[start:]
            imgfilenewname = imgfile.name[start:]
            newxmlfilename = os.path.join( savedir,  '_'.join(objnamelist) +  xmlfilenewname)
            newimgfilename = os.path.join( savedir,  '_'.join(objnamelist) +  imgfilenewname)
        copyfile(xmlfile,newxmlfilename )
        copyfile(imgfile, newimgfilename )

def main_split_dataset(filedir):
    ''' Devide dataset to several small datasets'''
    num = int(input(('Please input the number of Dataset will be splited:')))
    filelist,_ = getFiles(filedir,ImgType)
    num_file = len(filelist)
    condition = num_file//num
    for i in range(num):
        savedir = mkFolder(filedir, str(i))
        file_num = 0
        while file_num != condition:
            file = filelist.pop()
            move(file,savedir)
            for cpfile in findRelativeFiles(file):
                move(cpfile, savedir)
            file_num += 1
            if len(filelist) == 0:
                break


    imgfilefull, _ = getFiles(filedir, ImgType)

def main_add_figurelabel(filedir):
    ''' Add label for figure xml'''
    imgfiles, _ = getFiles(filedir, ImgType)
    label =  input("Label name:") 

    if label == "":
        label  = "temp"
        Min_x = 0
        Min_y = 0
        Max_x = 1
        Max_y = 1
    else:   
        Min_x = int(input("BOX Min_x:")) 
        Min_y = int(input("BOX Min_y:"))
        Max_x = int(input("BOX Max_x:"))
        Max_y = int(input("BOX Max_y:"))
 
    for imgfile in tqdm(imgfiles):
        xmlfile = imgfile[:-4] + '.xml'
        if os.path.exists(imgfile[:-4] + '.xml'):
            bboxlist,w,h = getObjectxml(xmlfile,classes='all')
        else:
            bboxlist,w,h = [],99999,99999
        bboxlist.append([Min_x,Min_y,Max_x,Max_y,0,label])
        xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": bboxlist}
        createObjxml(xmldic, imgfile[:-4] + '.xml', cls=[])

def main_movefilestoone(imgdir,num=0):
    """
     num = 0 move
     num > 0  rename file with dir.
     Move files into father dirctory"""
    for root, dirs, files in os.walk(imgdir):
        for file in tqdm(files):
            if num == 0:
                try:
                    move(os.path.join(root, file),imgdir)
                except:
                    print(file)
            else:

                folders = []
                file_path = root
                for i in range(num):
                    file_path, folder = os.path.split(file_path)
                    if folder != "":
                        folders.append(folder)
                    else:
                        if file_path != "":
                            folders.append(file_path)
                        break
                folders.reverse()
                folders.extend([file])

                foldesaname = "_".join(folders)
                copyfile(os.path.join(root, file),file_path+"//"+foldesaname)


def main_moveconfuse(imgdir):
    " Move error files into errorsamples folder"
    savedir = mkFolder(imgdir,'errorsamples')
    for line in open(imgdir+"errorlist.txt","r"):
        file = imgdir + line.split(' ')[0]
        for cpfile in findRelativeFiles(file):
            move(cpfile, savedir)

def main_mkdirforonedir(imgdir):
    "Move one image to folder named with image file name"
    _,imgfiles = getFiles(imgdir, ImgType)

    for id,img in enumerate(imgfiles):
        savedir = mkFolder(imgdir,str(id))
        move(os.path.join(imgdir,img), savedir)
def sparsebbox(bbox):
    label = bbox[0]
    x0 = bbox[1]
    y0 = bbox[2]
    w = bbox[3]
    h = bbox[4]
    return x0,y0,w,h

def saveCropObject(imgdir, file, savedir,xmlfile="crop.xml"):
    bbox, w, h = getObjectxml(imgdir+xmlfile, classes="all")
    img = cv2.imread(imgdir + file)
    for i,box in enumerate(bbox):
        x0,y0,w,h = sparsebbox(box)
        cv2.imwrite(str(Path(savedir,Path(file.replace(file[-4:],f"_{i}.jpg")))), img[x0:x0+w, y0:y0+h])

def saveCropObject(imgdir, file, savedir,num,wh):
    img = cv2.imread(imgdir + file)
    (h1,w1,_) = img.shape
    w,h = wh
    inter = int((w1-w) / num)


    for i in range(num):
        if i == 0:
            x0 = 0;y0 = 0;x1=w;y1 = h
        else:
            x0 = i * inter;
            y0 = 0
            x1 = i * inter+w;
            y1 = h
        cv2.imwrite(str(Path(savedir, Path(file.replace(file[-4:], f"_{i}.jpg")))), img[ y0:y1,x0:x1])

def main_cropfixedroi(imgdir,cls=["other"]):
    '''Move crop.xml file into the target dir.
       Crop imgs with xml，
       Note：Label Name: other
    '''
    num = int(input("Please input splited figures number"))

    savedir = mkFolder(imgdir,cls[0])
    _,imgfiles = getFiles(imgdir,ImgType)
    for file in tqdm(imgfiles):
        try:
            saveCropObject(imgdir, file,savedir,num,wh=[144,144])
        except Exception as e:
            print(e)
            print(traceback.format_exc())

def main_cropsquareroi(imgdir,clsname=["temp"]):
    "Move crop.xml file into the target dir."
    _,imgfiles = getFiles(imgdir,ImgType)
    for file in tqdm(imgfiles):
        try:
            for cls in clsname:
                savesquareImg(imgdir, file, cls,scale=0,square=False,fixroi =True)
        except Exception as e:
            print(e)
            print(traceback.format_exc())


def main_crop_square_image(imgdir,keepnoobj):
    '''
        Crop square imge, keep object in xml
        output : square img xml
    '''
    savedir = mkFolder(imgdir, "square_dataset")
    imgfull, imgfiles = getFiles(imgdir, ImgType)
    total = len(imgfiles)
    cls = input("Class you want to save(person,mask.Note: has the same sort as yaml): ")
    cls = cls.split(",")
    for id, file in enumerate(imgfiles):
        img = cv2.imread(imgdir + file)
        print("%d/%d Currrent image: %s" % (id + 1, total, file))
        bbox, w, h = getObjectxml(imgfull[id].replace(file[-4:], ".xml"), classes="all")
        #算图的位置
        s = min(w, h);
        l = max(w, h);
        n = int(np.round(w / h + 0.5)) if w > h else int(np.round(h / w + 0.5));
        start = np.linspace(0, l - s, n)

        for i in start:
            i = int(i)
            imgpath = str(savedir / str(file.replace(file[-4:], "_" + str(i) + file[-4:])))
            xmlpath = imgpath.replace(imgpath[-4:], ".xml")

            if w > h:
                # print(s*i-offset*(i+1),s*i-offset*(i+1)+s)
                cv2.imwrite(imgpath, img[0:s, i:i + s])
                window_xml(xmlpath, bbox, [i, 0, i + s, s], cls)
            else:
                cv2.imwrite(imgpath, img[i:i + s, 0:s], cls)
                window_xml(xmlpath, bbox, [0, i, s, i + s])

def main_movediffimg(imgdir):
    "Move different iamges by commparing files between img dirs"
    imgdir1 = input("Dir1:")+"\\"
    imgdir2 = input("Dir2:")+"\\"
    _, imgfiles1 = getFiles(imgdir1, ImgType)
    _, imgfiles2 = getFiles(imgdir2, ImgType)
    diff_files = mkFolder(imgdir1,"diff")
    for file in imgfiles2:
        if file  in imgfiles1:
            print (file)
            if os.path.exists(imgdir2+file):
                copyfile(os.path.join(imgdir2,file),os.path.join(diff_files,file))
            else:
                print(f'{file} is not found in dir 2！')
def main_mvimg2carmerafold(imgdir):
    "Move  images by to camera fold"
    img, imgname = getFiles(imgdir, ImgType)
    for i,name in enumerate(imgname):
        camera_num = name.split('_')[2]
        des = mkFolder(imgdir,str(camera_num))
        move(img[i],des)


def main_stretchfigure(imgdir):
    ""
    import numpy as np
    img, imgname = getFiles(imgdir, ImgType)
    des = mkFolder(imgdir, "res")
    w = 8192;h = 10000
    for i,name in enumerate(img):
        image = cv2.imread(name)
        h0,w0,_ = image.shape
        # 计算图像的均方差
        mean, std_dev = cv2.meanStdDev(image)
        # 提取标准差值

        gray_image = np.zeros(( h,w, 3), dtype=np.uint8)
        # 为每个通道分别赋值
        for channel in range(3):
            gray_image[:, :, channel] = int(mean[channel,0])  # 将当前通道的所有像素值设为0
        noise_image = np.random.normal(loc=0, scale=std_dev[0,0], size=(h,w,3))
        dest_image = gray_image + noise_image
        dest_image = cv2.GaussianBlur(dest_image,(17,17),11)
        x0 = random.randint(0,w-image.shape[0]);y0 = random.randint(0,h-image.shape[1])

        dest_image[y0:y0+h0, x0:x0+w0] = image

        # 提取标准差值
        cv2.imwrite(os.path.join(des,"res_"+str(i)+'.jpg'),dest_image)

def main_removeborder(dir,imsize=(1936,148)):
    '''
    Remove border and resize to user-defined size
    Args:
        files: image files:list
        imsize: resize size:tuple

    Returns:

    '''
    savedir = mkFolder(dir,"new")
    files,names = getFiles(dir,ImgType)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0,
                  cv2.IMWRITE_PNG_BILEVEL, 0,
                  cv2.IMWRITE_PNG_STRATEGY, 0,
                  cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 0]
    for file in tqdm(files):
        name = os.path.basename(file)
        im = cv2.imread(file)
        h, w, _ = im.shape
        row_indices = []
        col_indices = []
        for i in range(w):
            if calc_mode(im[:, i, 1])[1] / h > 0.9:
                col_indices.append(i)
        col_mask = np.ones((w), dtype=bool)
        col_mask[col_indices] = False
        im = im[:, col_mask]
        for i in range(h):
            if calc_mode(im[i, :, 1])[1] / w > 0.6:
                row_indices.append(i)
        row_mask = np.ones((h,), dtype=bool)
        row_mask[row_indices] = False
        im = im[row_mask]

        #
        im = cv2.resize(im, imsize)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # 设置PNG保存参数

        cv2.imwrite(os.path.join(savedir,name), im, png_params)

def main_rotate90_img(imgdir):
    '''
       Augmentation for Images
    '''
    imgfilespath,imgfiles= getFiles(imgdir,ImgType)
    savedir = mkFolder(imgdir, "rotate90")
    for id,imgfile in enumerate(imgfilespath):
        im = cv2.imread(imgfile)

        files = findRelativeFiles(imgfile)
        xmldir = imgdir + f"{imgfiles[id][:-4]}.xml"
        for file in files:
            if ".xml" in file:
                objectlist,w,h = getObjectxml(xmldir,classes='all')
        img, label = rotate90(im,targets=objectlist)
        im_h, im_w, _ = img.shape

        xmldic = {"size": {"w": str(im_w), "h": str(im_h), "c": str(3)}, "object": label}
        imgfile = str(savedir / imgfiles[id])
        cv2.imwrite(imgfile,img)
        createObjxml(xmldic, imgfile[:-4]+".xml", xmlfile=None)
    return 0

def main_samenamefile(dir):
    '''
        Move samme file name into one folder!
    '''
    format = input("Mov same name files into one folder(e.g. *.tif):")
    typelist = [i for i in format.split(",")]
    imgdirs, _ = getFiles(dir, typelist)
    samedir = mkFolder(dir,"same")

    for img in tqdm(imgdirs):
        files = findRelativeFiles(img)
        if len(files) > 1:
            # move(img, samedir)
            for file in files:
                move(file,samedir)


def main_compareimgdiff(dir):
    '''
        Move samme file name into one folder!
    '''
    format = input("Mov same name files into one folder(e.g. *.tif):")
    typelist = [i for i in format.split(",")]
    imgdirs, _ = getFiles(dir, typelist)
    res = []

    for img in tqdm(imgdirs):
        files = findRelativeFiles(img)
        if len(files) == 2:
            image1 = cv2.imread(files[0])
            image2 = cv2.imread(files[1])
            image1  = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2  = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            subtracted_image = cv2.absdiff(image1, image2)
            image1 = zscore(image1)
            image2 = zscore(image2)
            plt.imsave(files[0].replace(files[0][-4:],"_.jpg"),subtracted_image, cmap='gray')
            # 计算两幅图像的灰度直方图
            hist1,vector1 = calculate_histogram(image1)
            hist2,vector2 = calculate_histogram(image2)

            # 计算灰度直方图相似度
            similarity_hist,similarity_cosine = calculate_histogram_similarity(hist1, hist2,vector1,vector2)
            res.append([files[0],similarity_hist,similarity_cosine,np.max(subtracted_image)])
    res = pd.DataFrame(res)
    res.to_csv(dir+"res.csv")

def main_Huaatjsontoxml(datadir):
    typelist = ["*.json"]
    for root, dirs, files in os.walk(datadir):
        if dirs ==[]:
            jsondirs,_ = getFiles(root,typelist)
            if jsondirs == []:
                print(f"No json files: {root}")
            else:
                for jsonfile in jsondirs:
                    with open(jsonfile, 'r') as defectdic:
                        data = json.load(defectdic)
                    new_bbox = []
                    for defect in data["anno"]:
                        box = eval(defect["coor"])
                        defect_label = defect["label"][3:].lower()
                        new_xmin, new_ymin, new_xmax, new_ymax = box[0][0],box[0][1],box[1][0],box[1][1]
                        temp = [ new_xmin, new_ymin, new_xmax, new_ymax,1, defect_label]
                        new_bbox.append(temp)
                    print(jsonfile[:-8] + "jpg")
                    img = cv2.imread(jsonfile[:-8] + "jpg")
                    (h, w, _) = img.shape
                    xmldic = {"size": {"w": str(w), "h": str(h), "c": str(3)}, "object": new_bbox}
                    createObjxml(xmldic, jsonfile[:-4]+".xml", [])
    # print(data)
    return


def find_keys_by_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys

def main_S3CSV2XML(folddir):
    '''
    读入数据，解析成xml文件
    '''

    defect_dic = {
        "dqp": "HA",
        "qp": "HA",
        "bbqp": "HB",
        "jqtw": "HH",
        "clxtp": "IA",
        "zgxtp": "IB",
        "wdxtp": "IC",
        "bdxtp": "ID",
        "tph": "IF",
        "htp": "IG",
        "hb": "IK",
        "zh": "JJ",
        "hs": "KA",
        "cs": "KB",
        "zp": "KE",
        "dr": "KH",
        "kd": "KI",
        "bs": "SA",
        "bx": "SF",
        "jz": "HD",
        "dtdw": "E2",
        "dzyr": "LA",
        "zdxtp": "II",
        "yh": "JA",
        "nt": "IE",
        "zd": "NA",
        "hx": "HD",
        "bg": "WW"
    }

    csvfiles, _ = getFiles(folddir, ["*.csv"])
    for csvf in csvfiles:
        if os.path.exists(csvf[:-4]):
            print(csvf)
            db = pd.read_csv(csvf, encoding='gbk')
            db1 = db.iloc[1:]
            # db1= db1.loc[:, ['out_mat_no':'defectname','topx0':]]
            print(db1.keys())
            for i, d in db1.iterrows():
                jpgname = str(d["imagefile"])
                roix0 = str(d["roix0"])
                roiy0 = str(d["roiy0"])
                roix1 = str(d["roix1"])
                roiy1 = str(d["roiy1"])
                cls = str(d["defectclasscode"])
                keys = find_keys_by_value(defect_dic,  cls)
                bbox = [[roix0, roiy0, roix1, roiy1, 0, keys[0]]]
                print(bbox)
                xmldic = {"size": {"w": str(1536), "h": str(148), "c": str(3)}, "object": bbox}
                createObjxml(xmldic,  csvf[:-4] + "/" + jpgname, cls=[], xmlfile=None)
            # move(csvf, csvf[:-4])
        else:
            print("Error: no folder found! ")
            
def adjustw_ratio(x1,y1,x2,y2,ratio,imgw):
    w = x2-x1;h = y2-y1
    print(w/h)
    delta_w = int((h*ratio - w)/2)
    if x1 - delta_w < 0 :
        x2 = x2 + delta_w*2-x1;x1 = 0
    elif x2 + delta_w > imgw :
        x2 =  imgw ;x1 = imgw-delta_w*2
    else :
        x2 = x2+delta_w;x1 = x1-delta_w 
    w = x2-x1;h = y2-y1
    print(ratio,w/h)
        
    # #TODOLIST  
        
    # x2 = x2 + delta_w

    return x1,y1,x2,y2           
            
def adjustwh(x1,y1,x2,y2,thred):
    w = x2-x1;h=y2-y1
    ratio = int(max(w,h)/min(w,h))
    print(f"ratio:{ratio}")
    if ratio > thred:
        ratio = thred
        flag = True
        if w > h:
            delta_h = int((w/ratio-h)/2+1)
            if y1 - delta_h < 0:
                y1 = 0; y2= y2 + 2*delta_h
            else:
                y1 = y1-delta_h;y2=y2+delta_h

        else:
            delta_w = int((h/ratio-h)/2+1)
            if x1 - delta_w < 0:
                x1 = 0; x2= x2 + 2*delta_w
            else:
                x1 = x1-delta_w;x2=x2+delta_w
    else:
        flag = False
    return x1,y1,x2,y2,flag


def main_adjustwh(file_dir):
    """
    Adjust w h to a proper raio for Yolo label
    """
    xml_dirs, _ = getFiles(file_dir, LabelType)
    savedir = mkFolder(file_dir, "newxml")

    for xml in xml_dirs:
        bboxes, w, h = getObjectxml(xml, "all")
        new_bboxes = []
        flag_list = []
        for bbox in bboxes:
            # bbox[0], bbox[1], bbox[2], bbox[3], flag = adjustwh(bbox[0], bbox[1], bbox[2], bbox[3], thred=200)
            bbox[0], bbox[1], bbox[2], bbox[3]= adjustw_ratio(bbox[0], bbox[1], bbox[2], bbox[3],0.15,w)
            temp = [bbox[0], bbox[1], bbox[2], bbox[3], 1, bbox[5]]
            new_bboxes.append(temp)
            xmldic = {"size": {"w": str(w), "h": str(h), "c": "3"}, "object": new_bboxes}
            createObjxml(xmldic, os.path.join(savedir, os.path.basename(xml)))


def split_list_randomly(lst, num_parts):
    random.shuffle(lst)  # 随机打乱列表顺序
    avg = len(lst) // num_parts  # 计算每份的平均长度
    remainder = len(lst) % num_parts  # 计算剩余的元素个数

    result = []
    start = 0
    for i in range(num_parts):
        length = avg + 1 if i < remainder else avg  # 根据剩余元素个数决定每份的长度
        result.append(lst[start:start + length])
        start += length
    return result


def main_splitdataset(file_dir):
    """
    Random Move files to different folders.
    """
    img_dirs, _ = getFiles(file_dir, ImgType)
    n = int(input("Input the number you want to divide:"))
    group_list = split_list_randomly(img_dirs,n)
    for i,group in enumerate(group_list):
        savedir = mkFolder(file_dir,str(i))
        for f in group:
            for fs in findRelativeFiles(f):
                move(fs,savedir)

def main_removeduplicate(xmldir):
    """
    Remove duplicate "by" and adjust whbuy

    """
    _,xmlfiles = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    remDuplicateObjectxml(xmldir,xmlfiles,cls,isSavas=True)


def remDuplicateObjectxml(xmldir, xmlfiles, classes, isSavas=True):
    '''
    Description: remove object from xmlfile in VOC
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        xmldir[str]:xml file directory
        xmlfile[],classes
    Return:
        NaN
    Usage:
        filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/'
        xmlfiles = glob.glob(filedir + '*.xml')
        remObjectxml(filedir,xmlfiles,["person"],isSavas=False)
    '''

    # xmlfile = os.path.join(xmldir, xmlfile)
    savedir = mkFolder(xmldir, "rem_copy")
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        xmlpath = xmldir + xmlfile
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        objects = root.findall("object")
        isfindflag = 0
        seen = []
        for obj in objects:
            name = obj.find('name').text
            print(name)
            if name in classes:
                if name in seen:
                    root.remove(obj)
                else:
                    seen.append(name)
                isfindflag = 1

        if isfindflag == 1:
            print(xmlpath, os.path.join(savedir, xmlfile))
            copyfile(xmlpath, os.path.join(savedir, xmlfile))
            for cpfile in findRelativeFiles(xmlfile[:-4]):
                copyfile(cpfile, savedir)
            tree.write(xmlpath)

def main_adjustobjectxml(xmldir):

    """

    Args:
        file_dir:

    Returns:

    """
    xmlfiles,_ = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    adjustbbox(xmldir,xmlfiles,cls)

def adjustbbox(imgdir,xmlfiles,cls):

    os.chdir(os.path.dirname(imgdir))

    # 使用OpenCV读取图片

    for id, xmlfile  in enumerate(tqdm(xmlfiles)):

        bbox, w, h = getObjectxml(xmlfile,classes="all")
        bbox_list = []
        for i, bbox in enumerate(bbox):
            print(bbox[5]) 
            if bbox[5] in cls:

                b = (float(bbox[0])-10, float(0),float(bbox[2])+10,  float(h),0.5,bbox[5])
                print(b)
            else:
                b = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),bbox[5])
            bbox_list.append(b)
        xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": bbox_list}

        createObjxml(xmldic, xmlfile)

def main_masknonroi(imgdir):
    '''
    Mask nonroi with other color
    '''
    imgdirs,_ = getFiles(imgdir,ImgType)
    color = (255,0,0)
    savedir = mkFolder(imgdir,"newimg")
    masknonroi(imgdirs,savedir,color)



def masknonroi(imgdirs,savedir,color=(0,0,255)):

    for imgdir in tqdm(imgdirs):
        img,bboxes,w,h = preprocess(imgdir)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = color
        for i,bbox in enumerate(bboxes):
            xmin,ymin,xmax,ymax = bbox[1],bbox[2],bbox[3],bbox[4]
            bg[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(os.path.join(savedir,str(i)+"_" + os.path.basename(imgdir)),bg)

def main_movexmlverified(xmldir):
    '''
        Move file included object to object dir  
    '''
    savedir = mkFolder(xmldir,"verify")
    xmlfiles,_ = getFiles(xmldir,LabelType)
    chkVerifiedxml(xmlfiles,savedir)

def main_checkfigurequality(xmldir):
    '''
       Check Quality
    '''
    savedir = mkFolder(xmldir,"QualityFigure")
    imgfiles,_ = getFiles(xmldir,ImgType)
    chkFigurequality(imgfiles,savedir)

def chkFigurequality(imgfiles,savedir):
    for imgfile in imgfiles:
        image_path_encoded = imgfile.encode('utf-8').decode('utf-8')
        img = cv2.imread(image_path_encoded)
        red_channel = img[:, :, 2]  # 第三个通道是红色通道
        height, width, channels = img.shape
    
        # 计算红色通道最后五个像素的起始位置
        start_x = width - 5
        # 获取红色通道的最后五个像素的数值
        last_five_red_pixels = red_channel[-1, start_x:start_x+5]
        variance = np.var(last_five_red_pixels)
        print(variance)
        if variance == 0:
            for file in findRelativeFiles(imgfile):
                move(file,savedir)
        else:
            pass

def test():
    print("12312312")
    return 0

    

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"C:\Users\chaos\Desktop\new\image\temp/"
        # pass
    try:
        if action == "getFrame": # Extract frame from 
            print(main_extract_frame_from_video.__doc__)
            main_extract_frame_from_video(file_dir)
        elif action == "main_checkfigurequality":#main_checkfigurequality
            print(main_checkfigurequality.__doc__)
            main_checkfigurequality(file_dir)
        elif action =="main_movexmlverified":
            print(main_movexmlverified.__doc__)
            main_movexmlverified(file_dir)
        elif action =="main_masknonroi":
            print(main_masknonroi.__doc__)
            main_masknonroi(file_dir)
        elif action == "adjustobjectxml":
            print(main_adjustobjectxml.__doc__)
            # main_removeduplicate(file_dir)
            main_adjustobjectxml(file_dir)
        elif action == "splitdataset":
            print(main_splitdataset.__doc__)
            main_splitdataset(file_dir)
        elif action == "main_adjustwh":
            print(main_adjustwh.__doc__)
            main_adjustwh(file_dir)
        elif action =="S32XML":#
            print(main_samenamefile.__doc__)
            main_S3CSV2XML(file_dir)
        elif action == "main_Huaatjsontoxml":#Huaatjsontoxml
            main_Huaatjsontoxml(file_dir)
        elif action == "compare2img":#
            main_compareimgdiff(file_dir)
        elif action =="samenamefile":
            print(main_samenamefile.__doc__)
            main_samenamefile(file_dir)
        elif action == "main_remove_obj_from_xml":
            print(main_remove_obj_from_xml.__doc__)
            main_remove_obj_from_xml(file_dir)
        elif action == "voc2yolo":
            print(main_change_voc_to_yolo.__doc__)
            main_change_voc_to_yolo(file_dir)
        elif action == "main_change_cls_name":
            print(main_change_cls_name.__doc__)
            main_change_cls_name(file_dir)
        elif action == "changefilename":#changefilename
            print(main_rename_file_based_on_objection.__doc__)
            main_rename_file_based_on_objection(file_dir)
        elif action == 'splitYoloTrainVal':#splitYoloTrainVal
            print(main_yolo_train_val_set.__doc__)
            main_yolo_train_val_set(file_dir,task='trainval')
        elif action == "cropObject": #cropObject
            print(main_crop_object_img.__doc__)
            main_crop_object_img(file_dir)
        elif action == "plotBBox":#plotBBox
            print(main_plot_bbox.__doc__)
            main_plot_bbox(file_dir)
        elif action == "checklabelxml":#checklabelxml
            print(main_check_label_xml.__doc__)
            main_check_label_xml(file_dir)
        elif action == "squareimg":#squareimg
            print(main_create_square_image_samples.__doc__)
            main_create_square_image_samples_one_pic(file_dir)
        elif action == "plotinferres":
            print(main_plot_infer_res.__doc__)
            main_plot_infer_res(file_dir)
        elif action == "changeHSV":
            print(main_change_hsv.__doc__)
            main_change_hsv(file_dir)
        elif action == "clipsquareimage":#clipsquareimage
            print(main_change_hsv.__doc__)
            main_crop_square_image(file_dir)
        elif action == "changeYolo2Voc":
            print(main_change_yolo_to_voc.__doc__)
            main_change_yolo_to_voc(file_dir)
        elif action == "reduceVdieoFrame":
            print(main_video2video.__doc__)
            main_video2video(file_dir)
        elif action == "movObject":#movObject
            print(main_movobject.__doc__)
            main_movobject(file_dir)
        elif action == "remUnusedXML":
            print(main_remunusedfile.__doc__)
            main_remunusedfile(file_dir)
        elif action == "imagefiter":#imagefiter
            print(main_imagesize_filter.__doc__)
            main_imagesize_filter(file_dir)
        elif action == "splitimages":#splitimages
            print(main_split_images.__doc__)
            main_split_images(file_dir)
        elif action == "imgtovideo":
            print(main_img_to_video.__doc__)
            main_img_to_video(file_dir)
        elif action == "padding_image":#
            print(main_padding_image.__doc__)
            main_padding_image(file_dir)
        elif action == "resizeimage":#resizeimage
            print(main_resize_image.__doc__)
            main_resize_image(file_dir)
        elif action == "main_split_dataset":#
            print(main_split_dataset.__doc__)
            main_split_dataset(file_dir)
        elif action == "sobel_x":#
            print(main_change_sobelx.__doc__)
            main_change_sobelx(file_dir)
        elif action == "addfigurelabel":#addfigurelabel
            print(main_add_figurelabel.__doc__)
            main_add_figurelabel(file_dir)
        elif action == "movefilestoone":  #
            print(main_movefilestoone.__doc__)
            main_movefilestoone(file_dir)
        elif action == "moveerrorfiles":  # moveerrorfiles
            print(main_moveconfuse.__doc__)
            main_moveconfuse(file_dir)
        elif action == "mkdirforonedir":  #mkdirforonedir
            print(main_mkdirforonedir.__doc__)
            main_mkdirforonedir(file_dir)
        elif action == "movdiffimg":#movdiffimg"
            print(main_movediffimg.__doc__)
            main_movediffimg(file_dir)
        elif action == "mvimg2carmerafold":#
            print(main_mvimg2carmerafold.__doc__)
            main_mvimg2carmerafold(file_dir)
        elif action =="varscales":#varscales
            print(main_stretchfigure.__doc__)
            main_stretchfigure(file_dir)
        elif action == "cropfixedroi":#
            print(main_cropfixedroi.__doc__)
            main_cropfixedroi(file_dir)
        elif action == "imgchangetojpg":#
            print(main_imgchangetojpg.__doc__)
            main_imgchangetojpg(file_dir)
        elif action == "removeborder":#
            print(main_removeborder.__doc__)
            main_removeborder(file_dir)
        elif action == "rotate90img":
            print(main_rotate90_img.__doc__)
            main_rotate90_img(file_dir)
        elif action == "":
            print(main_movefilestoone.__doc__)
            main_movefilestoone(file_dir)
            main_movobject(file_dir,autoflag="?",numclass=99)
        elif action == "test":
            test()
            

    except Exception as e:
        print(e)
        print(traceback.format_exc())

    os.system("pause")
