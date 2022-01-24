from operator import truth
import os
import xmltodict
import glob
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from shutil import copyfile

from sklearn.model_selection import train_test_split

def renfile(filedir,format):
    '''
    Description:
        Rename file in filedir
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        filedir[str]:files directory
        format[str]: file format
    Return:
        NaN
    Usage:
        renfile(filedir,'.jpg')
    '''
    format = "*" + format
    files = glob.glob(filedir + format)
    for index,file in enumerate(files):
        try:
            os.rename(file,filedir + str(index)+format)
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        else:
            print('rename file success\r\n')

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

def getObjectxml(xmlfile,classes):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        xmlfile[str]: .xml file from labelimg
        classes[list]: Dest image
    Return:
        obj[list]: obeject list
    Usage:
        bboxlist = getObjectxml(xmlfile,classes)
    '''
    print ("Current process file:",xmlfile)
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

def VOC2Yolo(xmlfiles,classes):
    '''
    Description: Change xml to yolo format
    Author: Yujin Wang
    Date: 2022-01-22
    Args:
        xmlfiles[list]: xml files
        classes[list]: classes
    Return:
        NaN
    Usage:
        xmlfiles = glob.glob("./annotations" + '*.xml')
        classes = ["mask","nomask"]
        VOC2Yolo(xmlfiles,classes)
    '''
    for file in xmlfiles :
        file = file.replace("\\", "/")
        a = cv2.imread(file.replace(".xml",".jpg"))
        height, width, _ = cv2.imread(file.replace(".xml",".jpg")).shape
        out_file = open(file.replace(".xml",".txt"),'w') 
        bboxlist = getObjectxml(file,classes)
        for bbox in bboxlist:
                cls_id = classes.index(bbox["name"])
                b = (float(bbox["bndbox"]["xmin"]), float(bbox["bndbox"]["xmax"]), float(bbox["bndbox"]["ymin"]), float(bbox["bndbox"]["ymax"]))
                bb = convert((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
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

    filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/'
    xmlfiles = glob.glob(filedir + '*.xml')
    classes = ["person","alarm instrument"]
    VOC2Yolo(xmlfiles,classes)
    

    # filedir = r'D:\01_Project\01_Fangang\01_Ref_211232\1\images/copy/rem/'
    # xmlfiles = glob.glob(filedir + '*.xml')
    # xmlfiles = [i.replace("\\", "/") for i in xmlfiles]
    # classes = ["person"]
    # # remObjectxml(filedir,xmlfiles,classes)
    # for xmlfile in xmlfiles:
    #     remObjectxml(filedir,xmlfile,["person"])
    # classes = ["alarm instrument"]
    # VOC2Yolo(xmlfiles,classes)