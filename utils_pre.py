#!/usr/bin/env python
# coding:utf-8
import os
from stat import filemode
import sys
import time
import glob
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from utils_xml import *
from utils_math import *
from utils_cv import cv_show
from shutil import copyfile,move
from sklearn.model_selection import train_test_split
import random
import traceback

ImgType = ['*.jpg','*.jpeg','*.tif','*.png']
VideoType = ['*.avi','*.mp4']
LabelType = ['*.xml']

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
    savedir = Path(dir + foldername)
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
    for type in typelist:
        files.extend(glob.glob(folddir + type))
    files_wodir = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
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
                filename = file[:-4] + "_" + str(frametime).replace('.','p') + ".jpg"
                img_path = os.path.join(savedir ,filename)
                print (img_path)
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
        im = cv2.imread(imgfile)
        im_h,im_w,im_c = im.shape
        bbox = []
        print("%d/%d Currrent image: %s" % (id, total, imgfile))
        id += 1
        try:
            for line in open(imgfile[:-4]+'.txt'):

                clsid,cx,cy,w,h = [float(i) for i in line.split()]
                xmin,ymin,xmax,ymax =  xywh2xyxy([im_h,im_w],[cx,cy,w,h])
                bbox.append([xmin,ymin,xmax,ymax,0,clsid])
            xmldic = {"size": {"w": str(im_w), "h": str(im_h), "c": str(im_c)}, "object": bbox}
            createObjxml(xmldic, imgfile, cls=classes, xmlfile=None)

        except:
            print ("No yolo text file found!",imgfile)

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

    img = cv2.imread(imgpath)
    (h,w,_) = img.shape
    if h>w:
        return h,h//w,h%w,'w'
    if h<=w:
        return w,w//h,w%h,'v'

def createSquarImg(imgfiles):
    maxedge, ratio, padding, direction = getImgMaxLongEdge(imgfiles[0])
    num = len(imgfiles[1:])
    if direction == "v":
        padding = np.ones([int(padding/(num)), maxedge, 3], dtype=np.uint8)*255
    else:
        padding = np.ones([h, int(padding / (num)), 3], dtype=np.uint8) * 255
    img = cv2.imread(imgfiles[0])
    for img1 in imgfiles[1:]:
        img1 = cv2.imread(img1)
        if direction == "v":
            img = np.concatenate([img, padding], axis=0)
            img = np.concatenate([img, img1], axis=0)

        else:
            img = np.concatenate([img, padding], axis=1)
            img = np.concatenate([img, img1], axis=1)

    return img

def paddingSquare(img):
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


def saveCropImg(imgdir,imgfile,clsname,scale=0.1,square = True):
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
            scale = int((max(w,h)*(scale+1)-max(w,h))/2)
            offset = int(abs((h - w) / 2))
            if square == True:
                if h > w:
                    y1 = ymin - scale;y2 = ymax + scale;x1 = xmin - offset - scale; x2 = xmax + offset + scale
                    object = [[scale+ offset, scale , w + offset + scale, h + scale, 0, 0]]

                else:
                    y1 = ymin - offset - scale;y2 = ymax + offset + scale; x1 = xmin-scale;x2 = xmax+scale
                    object = [[scale, scale+offset, w+scale, h+offset+scale, 0, 0]]
            else:
                y1 = ymin-scale; y2 = ymax+scale; x1 = xmin-scale; x2 = xmax+scale

            ymin = max(0, y1);ymax = min(y2, height); xmin = max(0, x1);xmax = min(x2, width)
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w, c = crop_img.shape
            xmldic = {"size": {"w": str(w), "h": str(h), "c": str(c)},
                      "object": object}
            saveimg = os.path.join(savedir, imgfile[:-4] + '_' + clsname + '_' + str(id)+'.jpg')
            # h,w,c = crop_img.shape

            createObjxml(xmldic,saveimg,cls=[clsname])
            cv2.imwrite(saveimg,crop_img)
            
def plotRectBox(img,objectlist):
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

    for object in objectlist:
        xmin, ymin, xmax, ymax = object[1], object[2], object[3], object[4]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv2.putText(img, object[0], (int((xmin+xmax)/2),int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    return img
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
    tt =imglist[0]
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

def Video2Video(videofile,savedir,interval,offset):
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



def main_remove_obj_from_xml(xmldir): 
    '''
        Remove object from xml
    '''
    _,xmlfiles = getFiles(xmldir,LabelType)
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    remObjectxml(xmldir,xmlfiles,cls,isSavas=True)
 


def main_change_voc_to_yolo(xmldir):
    '''
        Change VOC to Yolo
    '''
    xmlfiles,_ = getFiles(xmldir,LabelType)
    cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
    # format = input("Please input image file format:")
    cls_name = cls_name.split(',')

    VOC2Yolo(xmlfiles,cls_name)

def main_change_yolo_to_voc(imgdir):
    '''
        Change  Yolo to Voc
    '''
    imgfiles,_ = getFiles(imgdir,ImgType)
    cls_name = input("Please input class you want(person,mask.Note: has the same sort as yaml):")
    cls_name = cls_name.split(',')
    Yolo2VOC(imgfiles,cls_name)

def main_change_cls_name(xmldir):
    '''
        Change class name
    '''
    xmlfiles = glob.glob(xmldir+ '*.xml')
    oldcls = input("Old class:")
    newcls = input("New class:")
    chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSavas=False)



def main_check_label_xml(xmldir):
    '''
        Check label xml
    '''
    xmlfiles,_ = getFiles(xmldir, LabelType)
    noobjectfiles = checkLabexml(xmlfiles)
    if len(noobjectfiles) != 0:
        savedir = mkFolder(xmldir,"noobject")
        for file in noobjectfiles:
            for cpfile in findRelativeFiles(file):
                move(cpfile,savedir)
    else:
        print("No unlabeled img found!")


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



def main_yolo_train_val_set(imgdir):
    '''
        Split train and val dataset
    '''
    img_serverdir = input("Train and validation img in serverdir(data/.../):")
    _,imgfiles = getFiles(imgdir,ImgType)
    imgfiles = [img_serverdir + i for i in imgfiles]
    test_size = float(input("Input the ratio of val:"))
    # print(imgfiles) 
    if test_size != 0.0:
        train_files, val_files = train_test_split(imgfiles, test_size=test_size, random_state=55)
        samplerdir = mkFolder(imgdir,'train_val')
        writeFile(samplerdir / 'train.txt', train_files)
        writeFile(samplerdir / 'val.txt',val_files)
    else:
        writeFile(imgdir + '/test.txt', imgfiles)


def main_crop_object_img(imgdir):
    '''
        Crop objet image
    '''
    clsname = input("Input class name:")
    scale = float(input("Input scale factor(max(h,w)):"))

    square = True if input("Crop image with padding(Y/N):") == "Y" else False

    clsname = clsname.split(',')
    _,imgfiles = getFiles(imgdir,ImgType)

    total = len(imgfiles)
    id = 1

    for file in imgfiles:
        print("%d/%d Currrent image: %s" %(id,total,file))
        try:
            for cls in clsname:
                saveCropImg(imgdir,file,cls,scale,square)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        id += 1


def main_plot_bbox(imgdir):
    '''
        Plot bbox in img
    '''
    savedir = mkFolder(imgdir,"plotbbox")
    imgfull,imgfiles = getFiles(imgdir,ImgType)
    cls = input("Class you want to plot(e.g. person,mask): ")
    cls = cls.split(",")
    total = len(imgfiles)
    for id,file in enumerate(imgfiles):
        bbox,_,_ = getObjectxml(imgfull[id].replace(file[-4:],".xml"),cls)
        img = cv2.imread(imgdir + file)
        img = plotRectBox(img,bbox)
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        imgpath = savedir / file
        cv2.imwrite(str(imgpath),img)
    return 

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
        
def main_clip_square_image(imgdir):
    savedir = mkFolder(imgdir,"square_dataset")
    imgfull,imgfiles = getFiles(imgdir,ImgType)
    total = len(imgfiles)
    cls = input("Class you want to save(person,mask.Note: has the same sort as yaml): ")
    cls = cls.split(",")
    for id,file in enumerate(imgfiles):
        print("%d/%d Currrent image: %s" %(id+1,total,file))
        bbox,w,h = getObjectxml(imgfull[id].replace(file[-4:],".xml"),classes="all")
        s = min(w,h);l = max(w,h) ;  n = int(np.round(w/h+0.5)) if w>h else int(np.round(h/w+0.5));
        offset = int(s - l/n)
        img = cv2.imread(imgdir + file)
        start = np.linspace(0,l-s,n)
        for i in start:
            i = int(i)
            imgpath = str(savedir / str(file.replace(file[-4:],"_"+str(i)+file[-4:])))
            xmlpath = imgpath.replace(imgpath[-4:], ".xml")

            if w > h:
                # print(s*i-offset*(i+1),s*i-offset*(i+1)+s)
                cv2.imwrite(imgpath,img[0:s,i:i+s])
                window_xml(xmlpath,bbox,[i,0,i+s,s],cls)
            else:
                cv2.imwrite(imgpath,img[i:i+s,0:s],cls)
                window_xml(xmlpath,bbox,[0,i,s,i+s])

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

def main_movobject(xmldir):
    '''
        Move file included object to object dir  
    '''
    xmlfiles,_ = getFiles(xmldir,LabelType)
    cls = input("Class name(only one label,e.g. person):")
    savedir = mkFolder(xmldir,cls)
    movObjectxml(xmldir,xmlfiles,cls,savedir)

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"D:\02_Study\01_PaddleDetection\Pytorch\yolov5\data\images/"
        file_dir = r"D:\01_Project\01_Pangang\08_Video\test0331\person_down\jianwei\test/"
        # pass
    if action == "getFrame":
        print(main_extract_frame_from_video.__doc__)
        main_extract_frame_from_video(file_dir)
    elif action == "remObj":
        print(main_remove_obj_from_xml.__doc__)
        main_remove_obj_from_xml(file_dir)
    elif action == "voc2yolo":
        print(main_change_voc_to_yolo.__doc__)
        main_change_voc_to_yolo(file_dir)
    elif action == "chgObjectxml":
        print(main_change_cls_name.__doc__)
        main_change_cls_name(file_dir)
    elif action == "changefilename":
        print(main_change_file_name.__doc__)
        main_change_file_name(file_dir)
    elif action == 'splitYoloTrainVal':
        print(main_yolo_train_val_set.__doc__)
        main_yolo_train_val_set(file_dir)
    elif action == "cropObject":
        print(main_crop_object_img.__doc__)
        main_crop_object_img(file_dir)
    elif action == "plotBBox":
        print(main_plot_bbox.__doc__)
        main_plot_bbox(file_dir)
    elif action == "checklabelxml":
        print(main_check_label_xml.__doc__)
        main_check_label_xml(file_dir)
    elif action == "squareimg":
        print(main_create_square_image_samples.__doc__)
        main_create_square_image_samples(file_dir)
    elif action == "plotinferres":
        print(main_plot_infer_res.__doc__)
        main_plot_infer_res(file_dir)
    elif action == "changeHSV":
        print(main_change_hsv.__doc__)
        main_change_hsv(file_dir)
    elif action == "clipsquareimage":
        print(main_change_hsv.__doc__)
        main_clip_square_image(file_dir)
    elif action == "changeYolo2Voc":
        print(main_change_yolo_to_voc.__doc__)
        main_change_yolo_to_voc(file_dir)
    elif action == "reduceVdieoFrame":
        print(main_video2video.__doc__) 
        main_video2video(file_dir)
    elif action == "movObject":
        print(main_movobject.__doc__) 
        main_movobject(file_dir)
    main_crop_object_img(file_dir)
    # main_extract_frame_from_video(file_dir)
    os.system("pause")