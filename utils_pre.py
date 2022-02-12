from operator import truth
import os
import sys
import xmltodict
import glob
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from shutil import copyfile

from sklearn.model_selection import train_test_split

def renfile(filedir,format,label,id=0):
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
        renfile(filedir,'.jpg')
    '''
    save_path = filedir+'out'+os.sep
    id = int(id)
    try:
        os.mkdir(save_path)
    except:
        pass
    files = glob.glob(filedir + format)
    for _,file in enumerate(files):
        str1 = file[:-3]+'*'
        duplicatefiles = glob.glob(str1)
        try:
            newname = save_path + str(id)+ '_' + label
            for file in duplicatefiles:
                copyfile(file,newname + file[-4:])
            print('%s is copied!',file)    
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        id += 1
        

def remObjectxml(xmldir,xmlfiles,classes,isSavas=False):
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
    if isSavas==True:
        save_path = xmldir + 'rem/'
        try:
            os.mkdir(save_path)
        except:
            pass
    # xmlfile = os.path.join(xmldir, xmlfile)
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        tree = ET.parse(os.path.join(xmldir,xmlfile))
        root = tree.getroot()
        objects = root.findall("object")
        for obj in objects:
            name = obj.find('name').text
            if name in classes:
                root.remove(obj)
        
        if isSavas==True:
            fn = xmlfile.replace("\\", "/").split("/")[-1].split(".json")[0]
            xmlfile = os.path.join(save_path,fn)
        print(xmlfile)
        tree.write(xmlfile)
    # return 0

def checkLabexml(xmldir,xmlfiles):
    '''
    Description:check label in xmlfile
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
    # print(xmldir,xmlfiles,oldcls,newcls)
    # xmlfile = os.path.join(xmldir, xmlfile)
    cls = {}
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        tree = ET.parse(os.path.join(xmldir,xmlfile))
        root = tree.getroot()
        objects = root.findall("object")
        for obj in objects:
            name = obj.find('name').text
            if name not in cls.keys():
                cls[name] = {"count":0,"files":[]}
            cls[name]["count"] += 1;cls[name]["files"].append(xmlfile)
    for name in cls.keys():
        print("Class name:%s\tNumber:%d" %(name,cls[name]["count"]))


def chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSavas=False):
    '''
    Description:Change class name in xmlfile
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
    # print(xmldir,xmlfiles,oldcls,newcls)
    if isSavas==True:
        save_path = xmldir + 'rem/'
        try:
            os.mkdir(save_path)
        except:
            pass
    # xmlfile = os.path.join(xmldir, xmlfile)
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        tree = ET.parse(os.path.join(xmldir,xmlfile))
        root = tree.getroot()
        objects = root.findall("object")
        for obj in objects:
            name = obj.find('name').text
            if name == oldcls:
                # root.remove(obj)
                # print(obj['name'])
                obj.find('name').text = newcls
        
        if isSavas==True:
            fn = xmlfile.replace("\\", "/").split("/")[-1].split(".json")[0]
            xmlfile = os.path.join(save_path,fn)
        print(xmlfile)
        tree.write(xmlfile)

def getObjectxml(xmlfile,classes):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        xmlfile[str]: .xml file from labelimg
        classes[list]: Class name
    Return:
        obj[list]: obeject list
    Usage:
        bboxlist = getObjectxml(xmlfile,classes)
    '''
    # print ("Current process file:",xmlfile)
    f = open(xmlfile,'rb')
    xmldict =  xmltodict.parse(f.read())
    obj = []
    try: 
        len(xmldict['annotation']["object"]) # Check "object" in keys
        try:
            # For multi-object
            for i in xmldict['annotation']["object"]:
                for object in classes: 
                    if i['name'] == object:
                        obj.append(i)
        except:
            # For one-object
            obj.append(xmldict['annotation']["object"])
            print ("Only one object is labeled!")   
    except:
        # No object
        print ("No object is found!")
    return obj

def getFrame(dir,flielist,interframe=100,startid=0,save_path=r"save_each_frames_front"):
    '''
    Description: Extract frame from video
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        dir[str]: video dir.
        flielist[list]:video list
        save_path[str]: frame save directory
    Return:
        NaN
    Usage:
        avi_list =  glob.glob(DocDir+".avi")
        filelist = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in avi_list[0]]
        print (filelist)
        getFrame(avi_list[0],filelist,save_path)
    '''
    save_path = dir + save_path
    try:
        os.mkdir(save_path)
    except:
        pass
    num = 0
    id = startid
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
        
        #set the start frame to read the video
        frameToStart = 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
        
        #get the frame rate
        rate = cap.get(cv2.CAP_PROP_FPS)
        print ("the frame rate is {} fps".format(rate))
        
        # get each frames and save
        frame_num = 0
        while True:
            
            ret, frame = cap.read()
            if ret != True:
                break

            if frame_num % interframe == 0:
                img_path = save_path +str(id)+'_' + str(num)+'_'+str(frame_num)+".jpg"
                print (img_path)
                cv2.imwrite(img_path,frame)
                id += 1
            frame_num = frame_num + 1
        
            # wait 10 ms and if get 'q' from keyboard  break the circle
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()

def VOC2Yolo(xmlfiles,classes='all',format='.jpg'):
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
    for file in xmlfiles :
        file = file.replace("\\", "/")
        # a = cv2.imread(file.replace(".xml",".jpg"))
        try:
            height, width, _ = cv2.imread(file.replace(".xml",format)).shape
        except:
            print("Image file cannot be readed by opencv!")
        out_file = open(file.replace(".xml",".txt"),'w') 
        bboxlist = getObjectxml(file,classes)
        for bbox in bboxlist:
                try:
                    cls_id = classes.index(bbox["name"])
                    b = (float(bbox["bndbox"]["xmin"]), float(bbox["bndbox"]["xmax"]), float(bbox["bndbox"]["ymin"]), float(bbox["bndbox"]["ymax"]))
                    bb = convert((width, height), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                except:
                    print("No object found in xml, file:%s" %(file))
        out_file.close()

def convert(size, box):
    '''
    Description: Change xyxy to xywh
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        size[list]: from img.shape
        box[list]:x1,y1,x2,y2
    Return:
        (x, y, w, h)
    Usage:
        height, width, _ = cv2.imread(file.replace(".xml",".jpg")).shape
        b = (float(bbox["bndbox"]["xmin"]), float(bbox["bndbox"]["xmax"]), float(bbox["bndbox"]["ymin"]), float(bbox["bndbox"]["ymax"]))
        bb = convert((width, height), b)
    '''
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def sampleset(filelist,dir,fn = 'train.txt'):
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
        sampleset(train_files,"./",fn = 'train.txt')
    '''
    save_path = dir
    try:
        os.mkdir(save_path)
    except:
        pass
    fd = dir +'/'+ fn
    print(fd)
    f = open(fd,'w')
    for i in filelist:
        f.write(i)
        f.write("\n")
    f.close()

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
    save_path = dir
    try:
        os.mkdir(save_path)
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
    save_path = file2dir
    try:
        os.mkdir(save_path)
    except:
        pass
    index = 0
    for file1 in filelist:
        file2xml = file2dir + str(index) + '.xml'
        file2jpg = file2dir + str(index) + '.jpg'
        copyfile(file1,file2xml)
        copyfile(file1.replace('.xml','.jpg'),file2jpg)
        index += 1

# MAIN PROCESSE
def main_extract_frame_from_video(videodir):
    '''
        Extract frame from video
    '''
    avi_list =  glob.glob(videodir+"*.avi")
    avi_list.extend(glob.glob(videodir+"*.mp4"))
    filelist = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in avi_list]
    interval = int(input("No. of interval frame:"))
    startid = int(input("Start ID:"))
    
    getFrame(videodir,filelist,interval,startid,r"out"+ os.sep)
    os.system("pause")


def main_remove_obj_from_xml(xmldir): 
    '''
        Remove object from xml
    '''
    xmlfiles = glob.glob(xmldir + '*.xml')
    cls = input("Class name(e.g.: person,mask):")
    cls = cls.split(',')
    remObjectxml(xmldir,xmlfiles,cls,isSavas=False)
    os.system("pause")


def main_change_voc_to_yolo(xmldir):
    '''
        Change VOC to Yolo
    '''
    xmlfiles = glob.glob(xmldir + '*.xml')
    cls_name = input("Please input class you want(Note: has the same sort as yaml):")
    format = input("Please input image file format:")
    cls_name = cls_name.split(',')
    VOC2Yolo(xmlfiles,cls_name,format)
    os.system("pause")



def main_change_cls_name(xmldir):
    '''
        Change class name
    '''
    xmlfiles = glob.glob(xmldir+ '*.xml')
    oldcls = input("Old class:")
    newcls = input("New class:")
    chgObjectxml(xmldir,xmlfiles,oldcls,newcls,isSavas=False)
    os.system("pause")


def main_check_label_xml(xmldir):
    '''
        Check label xml
    '''
    xmlfiles = glob.glob(xmldir+ '*.xml')
    checkLabexml(xmldir,xmlfiles)
    os.system("pause")

def main_change_file_name(xmldir):
    '''
        Rename files
    '''
    # xmlfiles = glob.glob(xmldir+ '*.xml')
    format = '*' + input("Input file format('.jpg'):")
    label = input("Add string in file name:")
    id = input("Start number:")
    renfile(xmldir,format,label,id)
    os.system("pause")


def main_yolo_train_val_set(imgdir):
    '''
        Split train and val
    '''
    img_serverdir = input("Train and validation img in serverdir:")
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    imgfiles = [img_serverdir+'/'+i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    # print(imgfiles) 
    train_files, val_files = train_test_split(imgfiles, test_size=0.1, random_state=55)
    savadir = imgdir+"/train_val/"
    sampleset(train_files,savadir ,fn = 'train.txt')
    sampleset(val_files,savadir ,fn = 'val.txt')
    os.system("pause")


if __name__ == "__main__":
    # sampleset_paddle
    # filedir = r'/data/wangyj/02_Study/Pytorch/dataset/mask/'
    # jpgfiles = glob.glob("./images/" + '*.jpg')
    # train_files, val_files = train_test_split(jpgfiles, test_size=0.1, random_state=55)
    # sampleset_paddle(train_files,"./",fn = 'train.txt')
    # sampleset_paddle(val_files,"./",fn = 'val.txt')
    #

    # filedir = "./annotations/"
    # xmlfiles = glob.glob(filedir + '*.xml')
    # xmlfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in xmlfiles]
    # for xmlfile in xmlfiles:
    #     remObjectxml(filedir,xmlfile,["person"])

    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'    
    # xmlfiles = glob.glob(filedir + '*.xml')
    # # xmlfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in xmlfiles]
    # # xmlfiles = xmlfiles[0]
    # # classes = ["alarm instrument"]
    # # VOC2Yolo(xmlfiles,classes)    
    # savecopy(xmlfiles,filedir+"copy/")



    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/'    
    # xmlfiles = glob.glob(filedir + '*.xml')
    # savecopy(xmlfiles,filedir+"copy/")
    
    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/' 
    # xmlfiles = glob.glob(filedir + '*.xml')
    # remObjectxml(filedir,xmlfiles,["person"],isSavas=False)

    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\jingbaoyi\images/copy/'
    # xmlfiles = glob.glob(filedir + '*.xml')
    # classes = ["person","alarm instrument"]
    # VOC2Yolo(xmlfiles,classes)
    

    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/rem/'
    # xmlfiles = glob.glob(filedir + '*.xml')
    # xmlfiles = [i.replace("\\", "/") for i in xmlfiles]
    # classes = ["person"]
    # # remObjectxml(filedir,xmlfiles,classes)
    # for xmlfile in xmlfiles:
    #     remObjectxml(filedir,xmlfile,["person"])
    # classes = ["alarm instrument"]
    # VOC2Yolo(xmlfiles,classes)

    # filedir = r"D:\01_Project\01_Fangang\01_Ref_211232\jingbaoyi\video\0124/"
    # # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\jingbaoyi\images/copy/'
    # avi_list =  glob.glob(filedir + '*.avi')
    # filelist = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in avi_list]
    # print (filelist)
    # getFrame(filedir,filelist)

    #renamefile
    # renfile(r"D:\01_Project\01_Fangang\05_Alarm\dataset\samples\images/",'.jpg',r"D:\01_Project\01_Fangang\05_Alarm\dataset\samples\images/out/")


    # extract frame
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        pass
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
    elif action == "checklabelxml":
        print(main_check_label_xml.__doc__)
        main_check_label_xml(file_dir)
    elif action == "changefilename":
        print(main_change_file_name.__doc__)
        main_change_file_name(file_dir)
    elif action == 'splitYoloTrainVal':
        print(main_yolo_train_val_set.__doc__)
        main_yolo_train_val_set(file_dir)