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
ImgType = ['*.jpg','*.jpeg','*.tif','*.png']
VideoType = ['*.avi','*.mp4']

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
 


def main_create_xml(imgdir,model = config["model"]["phone"]):
    '''
        Create xml by infering 
    '''
    infer = PersonInfer(model)
    _,imgfiles = getFiles(imgdir,ImgType)
    xmlfilefromobjdetect(infer,imgfiles,imgdir)

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"D:\01_Project\01_Pangang\08_Video\dataset\05_Test_video\alarm\test/"

    if action == "personxml":
        print(main_create_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["person"])
    elif action == "alarmxml":
        print(main_create_xml.__doc__)
        # main_create_xml(file_dir,model = config["model"]["alarm"])
        main_create_xml(file_dir,model = config["model"]["person_alarm"])
    elif action == "maskxml":
        print(main_create_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["mask"])
    elif action == "steelxml":
        print(main_create_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["steel"])
    elif action == "phonexml":
        print(main_create_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["phone"])
    elif action == "smokexml":
        print(main_create_xml.__doc__)
        main_create_xml(file_dir,model = config["model"]["smoke"])
    os.system("pause")