import json
import os
import sys
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
import cv2

pyscriptpath = r'D:\05_Trick\Trick\yolov5'
configpath = os.path.join(pyscriptpath,"config.json")
with open(configpath, 'r') as c:
    config = json.load(c)
sys.path.append(config["yolov5"])
from Detectbase.PersonInfer import PersonInfer




def create_node(tag, content=''):
    '''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag)
    element.text = content
    return element

def xywh2xyxy(*data):
    xc, yc, w, h = data[0],data[1],data[2],data[3]
    xmin = int(xc - w / 2)
    xmax = int(xc + w / 2)
    ymin = int(yc - h / 2)
    ymax = int(yc + h / 2)
    return xmin,xmax,ymin,ymax



def objxml(res,imgpath,clsid={"0":"person"},xmlfile='',isextend=False):

    if isextend == False:
        root = Element("annotation")
        root.append(create_node("filename","None"))
        tree = ElementTree(root)
    else:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
    for id,item in enumerate(res[0]):
        # xmin,xmax,ymin,ymax = xywh2xyxy(*item)

        item = item.tolist()
        xmin, ymin, xmax, ymax = item[0], item[1], item[2], item[3]
        key = str(int(item[-1]))
        if key in clsid.keys():
            obj = Element("object")
            obj.append(create_node("name",clsid[key]))
            obj.append(create_node("bndbox",''))
            obj[1].append(create_node("xmin",str(xmin)))
            obj[1].append(create_node("ymin",str(ymin)))
            obj[1].append(create_node("xmax",str(xmax)))
            obj[1].append(create_node("ymax",str(ymax)))
            root.append(obj)

    tree.write(imgpath.replace(imgpath[-4:],".xml"))

def xmlfilefromobjdetect(imglist,imgdir):   

    infer = PersonInfer(config["model"]["person"])
    total = len(imglist)
    for id,img in enumerate(imglist):
        imgpath = imgdir + img
        im = cv2.imread(imgpath)
        infer.run(im)
        res = infer.res
        objxml(res,imgpath)
        print("%d/%d,Current process image:%s, " %(id+1,total,img))
    os.system("pause")

def main_create_person_xml(imgdir):
    '''
        Create person xml
    '''
    imgfiles = glob.glob(imgdir + '*.jpg')
    imgfiles.extend(glob.glob(imgdir + '*.png'))
    imgfiles.extend(glob.glob(imgdir + '*.tif'))
    # print(imgfiles)
    imgfiles = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in imgfiles]
    xmlfilefromobjdetect(imgfiles,imgdir)

if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        pass

    if action == "personxml":
        print(main_create_person_xml.__doc__)
        main_create_person_xml(file_dir)