import json
import os
import sys
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
import cv2
from utils_pre import *
from utils_xml import *
from utils_math import *

pyscriptpath = r'D:\05_Trick\Trick'
configpath = os.path.join(pyscriptpath,"config.json")
with open(configpath, 'r') as c:
    config = json.load(c)
sys.path.append(config["yolov5"])
from Detectbase.PersonInfer import PersonInfer


def xmlfilefromobjdetect(infer,imglist,imgdir):   
    total = len(imglist)
    for id,img in enumerate(imglist):
        imgpath = imgdir + img
        im = cv2.imread(imgpath)
        h,w,c = im.shape
        infer.run(im)
        res = infer.res
        xmldic = {"size":{"w":str(w),"h":str(h),"c":str(c)},"object":res[0].tolist()}
        curxml = imgpath.replace(imgpath[-3:],'xml')
        if os.path.exists(curxml):
            createObjxml(xmldic,imgpath,infer.model.names,xmlfile=curxml)
        else:
            createObjxml(xmldic,imgpath,infer.model.names)
        print("%d/%d,Current process image:%s, " %(id+1,total,img))
 

def main_create_person_xml(imgdir):
    '''
        Create person xml
    '''
    infer = PersonInfer(config["model"]["person"])
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)

def main_create_mask_xml(imgdir):
    '''
        Create mask xml
    '''
    infer = PersonInfer(config["model"]["mask"])
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)
    

def main_create_alarm_xml(imgdir):
    '''
        Create alarm xml
    '''
    infer = PersonInfer(config["model"]["alarm"])
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)

def main_create_steel_xml(imgdir):
    '''
        Create steel xml
    '''
    infer = PersonInfer(config["model"]["steel"])
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)
    
def main_create_phone_xml(imgdir):
    '''
        Create phone xml
    '''
    infer = PersonInfer(config["model"]["phone"])
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)

def main_create_xml(imgdir,model = config["model"]["phone"]):
    '''
        Create xml by infering 
    '''
    infer = PersonInfer(model)
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(infer,imgfiles,imgdir)

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"D:\01_Project\01_Pangang\08_Video\train0303\0305\22\frame/"

    if action == "personxml":
        print(main_create_person_xml.__doc__)
        main_create_person_xml(file_dir)
    elif action == "alarmxml":
        print(main_create_alarm_xml.__doc__)
        main_create_alarm_xml(file_dir)
    elif action == "maskxml":
        print(main_create_mask_xml.__doc__)
        main_create_mask_xml(file_dir)
    elif action == "steelxml":
        print(main_create_steel_xml.__doc__)
        main_create_steel_xml(file_dir)
    elif action == "phonexml":
        print(main_create_phone_xml.__doc__)
        main_create_phone_xml(file_dir)
    elif action == "smokexml":
        print(main_create_phone_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["smoke"])
    os.system("pause")