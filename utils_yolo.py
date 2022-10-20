import json
import os
import sys
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
import cv2
from utils_pre import *
from utils_xml import *
import pandas as pd
from utils_math import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(10, 10)
import traceback

ImgType = ['*.jpg','*.jpeg','*.tif','*.png']
VideoType = ['*.avi','*.mp4']
pyscriptpath = os.path.split(os.path.realpath(__file__))[0]
configpath = os.path.join(pyscriptpath,"config.json")
with open(configpath, 'r') as c:
    config = json.load(c)

from Detectbase.Resnet_multiclass import ResnetDetector

def checkmodel(model,mode="detect"):
    sys.path.append(config[model['detect']])
    if model['detect'] == "yolov5":
        from Detectbase.YoloV5Infer import ObjectDetect, Validation
    elif model['detect'] == "yolov7":
        from Detectbase.YoloV7Infer import ObjectDetect, Validation
    elif model['detect'] == "yolov5-6.2":
        from Detectbase.YoloV5Infer import ObjectDetect, Validation
    if mode == "detect":
        return ObjectDetect(model)
    elif mode == "validation":
        return Validation(model)


def xmlfilefromobjdetect(infer,imglist,imgdir):
    '''
      Description: Create xml file from detection result
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
        infer model:
        imglist[list]
        imgdir[str]:
      Return:
         return
      Usage:
    '''
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
 
def check_confidence_conexit(bboxs,cls,conf):
    '''
      Description: Check confidence of object
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
         bbox[list]: object list.[['noalarm', 210, 828, 599, 1046], ['person', 204, 590, 631, 1520]]
         cls[list]:class list.['person', 'alarm', 'noalarm', 'alarm instrument']
         conf[list]:confidence list.[0.9.0.9]
      Return:
         checkres[dict]:{'person': False, 'alarm': False, 'noalarm': False, 'alarm instrument': False}
      Usage:
    '''
    res = [(False,0) for i in range(len(cls))]
    checkres = dict(zip(cls,res))
    for bbox in bboxs:
        clsname = bbox[0]; clsconf = bbox[5]
        conf_threshold = conf[cls.index(clsname)]
        if clsname in cls:
            if clsconf > conf_threshold:
                checkres[clsname] = (True,clsconf)
        else:
            pass
    return checkres


def count(d):
    '''
      Description: Get len and dictionary
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
         d: dictionary
      Return:
         return[int]:max length of dictionary
      Usage:
    '''
    if isinstance(d, dict):
        return max(count(v) if isinstance(v, dict) else 0 for v in d.values()) + 1
    else:
        return 0

def decison(ret,checkdict,cases = ["Background","True","False"]):
    '''
      Description: Make decison based on the rules
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
         ret[dict]: xml bbounding box information.
         checkdict[dict]:rules in config
      Return:
          Boolaan
      Usage:
    '''
    loop = count(checkdict)
    for i in range(loop):
        klist = list(checkdict.keys())

        checkcount = 0
        for k in klist:
            # print(type(ret[k]),type(checkdict[k]))
            if ret[k] == False:
                checkcount += 1
                if checkcount == len(klist):
                    if i != loop-1:
                        return None                       # BG(ignore)
                    else:
                        return "bg"                       # False
            elif ret[k] == checkdict[k]:
                return k                               # True
            else:
                if isinstance(checkdict[k],dict):
                    checkdict = checkdict[k]
                else:
                    pass
    return False

def decison_cls(ret,checkdict,case):
    '''
      Description: Make decison based on the rules
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
         ret[dict]: xml bbounding box information.
         checkdict[dict]:rules in config
      Return:
          Boolaan
      Usage:
    '''
    check_res = []
    max_conf = 0
    max_k = 'bg'
    for k,v in ret.items():
        if v[0] == True and checkdict[k] == True:
            check_res.append(k)
            if v[1] > max_conf:
                max_k  = k
                max_conf = v[1]
        if case in check_res:
            return case
    return max_k


def DOE_FullFactor(dimensons,checkrange):
    '''
      Description: Design of experiment_ FullFactors
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
          dimensons[int]:Number of parameters
          checkrange[list]:[[start,end,interval number]] ref to np.linespace()
      Return:
          ret[list]:A full factor iterator
      Usage:
    '''
    import itertools
    data = []
    for i in range(dimensons):
        data.append(np.linspace(checkrange[i][0],checkrange[i][1],checkrange[i][2]))
    ret = list(itertools.product(*data))
    return ret

def checkPerformance(cases,xmldir,checkdict,cls,conf):
    '''
      Description: Check model performance based on rules and confidence
      Author: Yujin Wang
      Date: 2022-04-11
      Args:
            cases[list]["True","False"]: Classified cases
            xmldir[str]:xml directory
            checkdict[dict]：rules
            cls[dict]:Object for check performanc, must be included in rules
            conf[dict]:Object confidence threshold
      Return:
          res[list]:file,actural,prediction
      Usage:
    '''
    res = []
    for case in cases:
        casedir = os.path.join(xmldir, case)
        xmlfiles, _ = getFiles(casedir, LabelType)
        id = 0
        for xmlfile in xmlfiles:  # check xml file
            id += 1
            bboxs, _, _ = getObjectxml(xmlfile, cls)
            filterbox = check_confidence_conexit(bboxs, cls, conf)  #
            dec = decison_cls(filterbox, checkdict,case)
            res.append([xmlfile, case, str(dec)])

    return res



def cm_plot(cases,conf,y_act, y_pred,imgdir,labels,plotflag = True):
    '''
    y: 真实值
    y_pred:预测值
    '''
    '''混淆矩阵绘画'''
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.metrics import PrecisionRecallDisplay
    plt.rcParams['font.size'] = 7
    cm = confusion_matrix(y_act, y_pred,labels = cases ) # 混淆矩阵
    a = [np.sum(i) for i in cm]
    # cm_row = cm / np.sum(cm, axis=1)
    # cm_col = cm / np.sum(cm, axis=0)
    accuracy = accuracy_score(y_act, y_pred)
    precision = precision_score(y_act, y_pred,average='weighted')
    recall = recall_score(y_act, y_pred,average='weighted')
    f1 = f1_score(y_act, y_pred,average='weighted')

    if plotflag:

        plt.matshow(cm, cmap=plt.cm.Greens)
        plt.colorbar()  # 颜色标签
        for x in range(len(cm)):  # 数据标签
            for y in range(len(cm)):
                row = round(cm[x, y]/a[x], 2);
                col = round(cm[x, y], 2)
                # info = f"Recall:{row}\nPrecison:{col}"
                info = f"{row}"
                plt.annotate(info, xy=(y, x), verticalalignment='center', horizontalalignment='center')
        plt.xlabel('Predicted label')  # 坐标轴标签
        plt.ylabel('True label')  # 坐标轴标签
        # labels=["BG","False","True"]
        plt.yticks(range(len(labels)), labels)
        plt.xticks(range(len(labels)), labels)

        conf = [round(i,2) for i in conf]
        plt.title("{}\nA:{} P:{} R:{} F1:{} Recall_33:{} Precion_33:{}\n".format(conf,round(accuracy,2),round(precision,2),round(recall,2),round(f1,2),round(row,2),round(col,2)))

        plt.savefig(imgdir+"/Confuse_Matrix.jpg",dpi=1200)
    return {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1}

# main program


def main_create_xml(imgdir,model = config["model"]["phone"]):
    '''
        Create xml by infering
    '''


    infer = checkmodel(model)
    _,imgfiles = getFiles(imgdir,ImgType)
    xmlfilefromobjdetect(infer,imgfiles,imgdir)



def main_split2class(xmldir,rules):
    '''
        Move figs to class fold
    '''
    cls = rules['name']
    conf = rules['confidence']
    cases = rules["cases"]
    checkdict = rules['priority']
    xmlfiles, _ = getFiles(xmldir, LabelType)
    savedir_bg = mkFolder(xmldir, cases[0])
    savedir_true = mkFolder(xmldir,cases[1])
    savedir_false = mkFolder(xmldir, cases[2])
    total = len(xmlfiles)
    id = 0
    for xmlfile in xmlfiles:
        id += 1
        print("%d/%d,Current process image:%s" % (id, total, xmlfile))
        try:
            bboxs, _, _ = getObjectxml(xmlfile, cls)
            filterbox = check_confidence_conexit(bboxs, cls, conf)
            dec = decison(filterbox, checkdict,cases = cases)
            if dec == cases[0]:
                for cpfile in findRelativeFiles(xmlfile):
                    move(cpfile, savedir_bg)
            elif dec == cases[1]:
                for cpfile in findRelativeFiles(xmlfile):
                    move(cpfile, savedir_true)
            elif dec == cases[2]:
                for cpfile in findRelativeFiles(xmlfile):
                    move(cpfile, savedir_false)
        except Exception as e:
            print(e)
            print(traceback.format_exc())


# def comaparepred(confs,)

def main_checkConfidence(xmldir,rules,DOE = None):
    '''
        Check object confidence
        1. Run infer to get confidence it will be write in xml files.(Latest version)
        2. Please define the parameters in config.json "rules", if run "DOE", DOE parameters must be defined.
        3. Split xmlfiles to 2 folders, "True" and "False"
        4. Run this program will get confuse matrix and csv
    '''
    cls = rules['name']
    checkdict = rules['priority']
    cases = rules["cases"]
    if DOE:
        checkrange = [DOE["parameter"][key] for key in DOE["parameter"].keys()]
        dimensons = len(checkrange)
        confs = DOE_FullFactor(dimensons, checkrange)
        plotflag = False
        resfile = "/resdata_DOE.csv"
    else:
        confs = [rules['confidence']]
        plotflag = True
        resfile = "/resdata.csv"
    report = []
    total = len(confs)
    maxvalue = 0
    best = []

    for id,conf in enumerate(confs): # Loopover all parameters
        print("{}/{}:{}".format(id+1,total,conf))
        res = checkPerformance(cases,xmldir,checkdict,cls,conf)                    # Judge the class
        res = pd.DataFrame(res, columns=["File", "Actual", "Prediction"])
        labels = []
        eva = cm_plot(cases,conf,res['Actual'], res["Prediction"], xmldir,cls , plotflag = plotflag)
        temp = list(conf)
        temp1 = list(eva.values())
        temp.extend(temp1)
        report.append(temp)
        if DOE and eva[DOE["object"]] > maxvalue :
            maxvalue = eva[DOE["object"]]
            print(maxvalue)
            best = conf
    temp = list(eva.keys())
    cls.extend(temp)
    resdata = pd.DataFrame(report,columns=cls)
    resdata = resdata.dropna()
    resdata.to_csv(xmldir+resfile)
    if DOE:
        print(f"Best res:{best}")
        res = checkPerformance(cases, xmldir, checkdict, cls, best)
        res = pd.DataFrame(res, columns=["File", "Actual", "Prediction"])
        cm_plot(cases, best, res['Actual'], res["Prediction"], xmldir, plotflag=True)

    res.to_csv(xmldir + "Actual_prediction.csv")
    
def main_val_xml(imgdir,model = config["model"]["phone"]):
    '''
        Create xml by infering
    '''
    ptname = model['weights'].split('/')[-1] + "_" + str(model["imgsize"])
    save_dir = mkFolder(imgdir, "validation_"+ptname)
    val = checkmodel(model,mode="validation")
    val.run(imgdir,str(save_dir))


def moveimgfile(infer,imgfiles,imgdir,classes):
    savedirs = []
    for cls in classes:
        savedirs.append(mkFolder(imgdir, cls))

    for im in imgfiles:
        res = infer.predict(im)
        move(im,savedirs[res])


def main_classfy(imgdir,model = config["model"]["phone_3cls"]):
    '''
    Resnet for classifying images
    '''
    infer = ResnetDetector(model)
    imgfiles,_ = getFiles(imgdir,ImgType)
    moveimgfile(infer,imgfiles,imgdir,model['classes'])


if __name__ == "__main__":
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
        file_dir = r"D:\02_Project\02_Baosteel\01_Hot_rolling_strip_steel_surface_defect_detection\03_\temp/"
    try:
        if action == "personxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["person"])
        elif action == "alarmxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["alarm"])
            # main_create_xml(file_dir,model = config["model"]["person_alarm"])
        elif action == "maskxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["mask"])
        elif action == "mask_yl_xml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["mask_yl"])
        elif action == "steelxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["steel"])
        elif action == "alarm4clsxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["alarm4cls"])
        elif action == "phonexml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["phone"])
        elif action == "smokexml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["smoke"])
        elif action == "baosteel_surfacedefect_20cls":#baosteel_surfacedefect_20cls
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["baosteel_surfacedefect_20cls"])
        elif action == "baosteel_surfacedefect_20cls_Yolov7":  # baosteel_surfacedefect_20cls_Yolov7
            print(main_create_xml.__doc__)
            main_create_xml(file_dir, model=config["model"]["baosteel_surfacedefect_20cls_Yolov7"])
        elif action == "helmetxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["helmet"])
        elif action == "safetybeltxml":
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["safetybelt"])
        # main_create_xml(file_dir, model=config["model"]["person_alarm"])
        elif action == "mask_9clstxml":#mask_9clstxml
            print(main_create_xml.__doc__)
            main_create_xml(file_dir,model = config["model"]["mask_9cls"])
        elif action == "checkconfidence":#checkconfidence
            print(main_checkConfidence.__doc__)
            name = input("Object rules for check(alarm,mask):")
            main_checkConfidence(file_dir,rules=config["rules"][name])
        elif action == "moveimage2class":
            print(main_split2class.__doc__)
            name = input("Object rules for check(alarm,mask):")
            main_split2class(file_dir,rules=config["rules"][name])
        elif action == "DOEConfidence":
            print(main_checkConfidence.__doc__)
            name = input("Object rules for check(alarm,mask):")
            main_checkConfidence(file_dir,rules=config["rules"][name],DOE=config["DOE"][name])
        elif action == "validation":#validation
            name = input("Validation for yolov5(alarm,mask):")
            main_change_voc_to_yolo(file_dir,cls=config["model"][name]["classes"])
            main_yolo_train_val_set(file_dir, task='test')
            main_val_xml(file_dir, model=config["model"][name])
        elif action == "phone_falldown_5cls":#classfication
            print(main_classfy.__doc__)
            print("phone_falldown_5cls")
            main_classfy(file_dir, model=config["model"]["phone_falldown_5cls"])
        elif action == "falldown_2cls":#classfication
            print(main_classfy.__doc__)
            print("falldown_2cls")
        # main_classfy(file_dir, model=config["model"]["falldown_2cls"])
        # main_create_xml(file_dir, model=config["model"]["person"])
        # main_create_xml(file_dir, model=config["model"]["person"])
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    os.system("pause")
