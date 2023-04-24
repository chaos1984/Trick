import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
from utils_pre import *

def indent(elem, level=0):
    '''xml 对齐
       elem:节点
       level:级别 默认0
    '''
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
def create_node(tag, content=''):
    '''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag)
    element.text = content
    element.tail = '\n'
    return element

def combineXMLinDirection(xmlfilelist, edgelen,fignum,padding,direction,flip):
    length = (edgelen - padding) / fignum
    offest = length + padding / (fignum - 1)

    if direction == 'v':
        return combineXML(xmlfilelist,flip , 0, offest)
    else:
        return combineXML(xmlfilelist,flip , offest, 0)
    
def combineXML(xmlfilelist,flip,xoffset=0,yoffset=0):
    tree = ET.parse(xmlfilelist[0])
    root = tree.getroot()
    root.findall("size")[0].findall("width")[0].text = root.findall("size")[0].findall("width")[0].text = str(max(int(root.findall("size")[0].findall("width")[0].text) , int(root.findall("size")[0].findall("height")[0].text)))
    # size = root.findall("size")

    for i,xmlfile in enumerate(xmlfilelist[1:]):
        objectlist,imgw,imgh = getObjectxml(xmlfile,classes='all')
        xc = imgw/2;yc = imgh/2

        for object in objectlist:
            obj = Element("object")
            obj.append(create_node("name", object[0]))
            obj.append(create_node("bndbox", ''))
            if flip[i] == "o":
                xmin = object[1];ymin = object[2];xmax = object[3] ;ymax = object[4];
            elif flip[i] == "vh":
                xmax = xc + abs(xc - object[1]) if xc > object[1] else xc - abs(xc - object[1]);
                ymax = yc + abs(yc - object[2]) if yc > object[2] else yc - abs(yc - object[2]);
                xmin = xc + abs(xc - object[3]) if xc > object[3] else xc - abs(xc - object[3]);
                ymin = yc + abs(yc - object[4]) if yc > object[4] else yc - abs(yc - object[4]);
            elif flip[i] == "v":
                xmin = object[1];
                ymin = imgh - object[4];
                xmax = object[3];
                ymax = imgh - object[2];
            elif flip[i] == "h":
                xmin = imgw - object[3];
                ymin = object[2];
                xmax = imgw - object[1];
                ymax = object[4];
            if xmax < xmin or ymax < ymin:
                print(f"error:{xmlfile}，{flip[i]},{object[0]}")
            obj[1].append(create_node("xmin", str(xmin+xoffset*(i+1))))
            obj[1].append(create_node("ymin", str(ymin+yoffset*(i+1))))
            obj[1].append(create_node("xmax", str(xmax+xoffset*(i+1))))
            obj[1].append(create_node("ymax", str(ymax+yoffset*(i+1))))
            root.append(obj)
    for elem in root:
        indent(elem, level=0)
    return tree

def createObjxml(res,imgpath,cls=[],xmlfile=None):
    '''
    Description: Make a object user defined xml file 
    Author: Yujin Wang
    Date: 2022-02-14
    Args:
        res[list]:[[],[]....]
        imgpath[str]:img path
        cls[dict]: key: clsid; value: clsname
        xmlfile[str]:xml file path
    Return:
        write a xml file to record yolo res.
    Usage:
    '''
    # print(res,imgpath)
    if xmlfile==None: #creat new xml
        root = create_node("annotation","")
        root.append(create_node("folder", "None"))
        root.append(create_node("filename","None"))
        root.append(create_node("path","None"))
        source = create_node("source","")
        source.append(create_node("database","Unknown"))
        root.append(source)
        size = create_node("size","")
        size.append(create_node("width",res["size"]["w"]))
        size.append(create_node("height",res["size"]["h"]))
        size.append(create_node("depth",res["size"]["c"]))
        root.append(size)
        #
        tree = ElementTree(root)
    else: # xml
        tree = ET.parse(xmlfile)
        root = tree.getroot()
    for id,item in enumerate(res["object"]):
        # xmin,xmax,ymin,ymax = xywh2xyxy(*item)
        if type(item[0]) == str:
            temp = item[0]
            item.pop(0)
            item.append(temp)
        try:
            xmin, ymin, xmax, ymax,confidence = int(item[0]), int(item[1]), int(item[2]), int(item[3]), float(item[4])
        except:
            xmin, ymin, xmax, ymax, confidence = int(item[0]), int(item[1]), int(item[2]), int(item[3]), 0
        obj = create_node("object","")
        if cls != []: #if label is num check dictionary
            obj.append(create_node("name",cls[int(item[-1])]))
        else:   #if label is str, no check
            obj.append(create_node("name", item[-1]))
        obj.append(create_node("pose",'Unspecified'))
        obj.append(create_node("truncated",'0'))
        obj.append(create_node("difficult",'0'))
        obj.append(create_node("bndbox",''))
        obj[4].append(create_node("xmin",str(max(xmin,0))))
        obj[4].append(create_node("ymin",str(max(ymin,0))))
        obj[4].append(create_node("xmax",str(min(xmax,int(res["size"]["w"])))))
        obj[4].append(create_node("ymax",str(min(ymax,int(res["size"]["h"])))))
        obj[4].append(create_node("confidence", str(confidence)))
        root.append(obj)
    for elem in root:
        indent(elem, level=0)
        format_index = imgpath.index('.')
    tree.write(imgpath[:format_index ]+".xml")


def movObjectxml(xmlfiles,cls,savedir,numclass = 99):
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
    
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")

        tree = ET.parse(xmlfile)
        root = tree.getroot()
        objects = root.findall("object")
        if len(objects) <= numclass:
            for obj in objects:
                name = obj.find('name').text
                if name.strip() == cls:
                    for cpfile in findRelativeFiles(xmlfile[:-4]):
                        move(cpfile,savedir)
        else:
            continue
                

def remObjectxml(xmldir,xmlfiles,classes,isSavas=True):
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
    savedir = mkFolder(xmldir,"rem_copy")
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        xmlpath = xmldir+xmlfile
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        objects = root.findall("object")
        isfindflag = 0
        for obj in objects:
            name = obj.find('name').text
            if name in classes:
                root.remove(obj)
                isfindflag = 1


        if isfindflag == 1:
            print(xmlpath,os.path.join(savedir,xmlfile))
            copyfile(xmlpath,os.path.join(savedir,xmlfile))
            for cpfile in findRelativeFiles(xmlfile[:-4]):
                copyfile(cpfile,savedir)
            tree.write(xmlpath)


def checkLabexml(xmlfiles):
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
    noobject = []
    for xmlfile in xmlfiles:
        # file = file.replace("\\", "/")
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        objects = root.findall("object")
        if len(objects) == 0:
            print("No object found in img: %s" %(xmlfile))
            noobject.append(xmlfile)
        for obj in objects:
            name = obj.find('name').text
            if name not in cls.keys():
                cls[name] = {"count":0,"files":[],"confidence":[],"area":[]}
            cls[name]["count"] += 1;cls[name]["files"].append(xmlfile)
            bndbox = obj.find('bndbox')
            try:
                confidence = round(float(bndbox.find('confidence').text),3)
                xmin,ymin,xmax,ymax = int(bndbox.find('xmin').text),int(bndbox.find('ymin').text),int(bndbox.find('xmax').text),int(bndbox.find('ymax').text)
                cls[name]["confidence"].append(confidence)
                cls[name]["area"].append((ymax-ymin)*(xmax-xmin))
            except:
                print(f"Error xmlfile:{xmlfile}")
    for name in cls.keys():
        print("Class name:%s\tNumber:%d" %(name,cls[name]["count"]))
    return noobject,cls

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
        savedir = mkFolder(dir,'rem')
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
            xmlfile = os.path.join(savedir,fn)
        print(xmlfile)
        tree.write(xmlfile)

def flipObjextxml(xmlfile,augfiledir,fliptype="v") : #0-hflip;1-vflip;-1-hvflip
    objectlist,w,h = getObjectxml(xmlfile,classes='all')
    xc = int(w/2); yc = int(h/2)
    new_objectlist = []
    for object in objectlist:
        if fliptype == "v" :
            new_object = [object[1],h-object[2],object[3],h-object[4],1.0,object[0]]
        elif fliptype == "h" :
            new_object = [w-object[1],object[2],w-object[3],object[4],1.0,object[0]]
        elif fliptype == "vh" :
            new_object = [w-object[1],h-object[2],w-object[3],h-object[4],1.0,object[0]]
        new_objectlist.append(new_object)
    xmldic = {"size": {"w": str(w), "h": str(h), "c": '3'}, "object": new_objectlist}
    # createObjxml(res,imgpath,cls=[],xmlfile=None)(new_objectlist)
    createObjxml(xmldic, augfiledir, cls=[], xmlfile=None)

def getObjectxml(xmlfile,classes):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        xmlfile[str]: .xml file from labelimg
        classes[list]: Class name
    Return:
        obj[list]: obeject list,[['person', 592, 657, 726, 1077],['person', 592, 657, 726, 1077]]
        w:img width
        h:img height
    Usage:
        bboxlist = getObjectxml(xmlfile,classes)
    '''
    # print ("Current process file:",xmlfile)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    size = root.findall("size")
    w = int(size[0].find("width").text); h = int(size[0].find("height").text)
    objects = root.findall("object")
    objectlist = []

    if len(objects) != 0:
        for obj in objects:
            name = obj.find('name').text
            if name in classes or classes == "all":
                bndbox = obj.find('bndbox')
                box = [name]

                for child in bndbox:
                    if len(child.text.split(".")) != 2:
                        box.append(int(child.text))
                    else:
                        box.append(float(child.text))
                if len(box) < 6:
                    box.append(0) #confidence
                objectlist.append(box)
    else:
        pass
    return objectlist,w,h


def read_xml(in_path):
    '''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree
 
def write_xml(tree, out_path):
    '''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8",xml_declaration=True)
 
def if_match(node, kv_map):
    '''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True
 
#---------------search -----
 
def find_nodes(tree, path):
    '''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)
 
 
def get_node_by_keyvalue(nodelist, kv_map):
    '''根据属性及属性值定位符合的节点，返回节点
       nodelist: 节点列表
       kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes
 
#---------------change -----
 
def change_node_properties(nodelist, kv_map, is_delete=False):
    '''修改/增加 /删除 节点的属性及属性值
       nodelist: 节点列表
       kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete: 
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))
            
def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''改变/增加/删除一个节点的文本
       nodelist:节点列表
       text : 更新后的文本'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text
            

def add_child_node(nodelist, element):
    '''给一个节点添加子节点
       nodelist: 节点列表
       element: 子节点'''
    for node in nodelist:
        node.append(element)
        
def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    '''同过属性及属性值定位一个节点，并删除之
       nodelist: 父节点列表
       tag:子节点标签
       kv_map: 属性及属性值列表'''
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)
                        
 
 
if __name__ == "__main__":
    
    #1. 读取xml文件
    tree = read_xml("./test.xml")
    
    #2. 属性修改
      #A. 找到父节点
    nodes = find_nodes(tree, "processers/processer")
      #B. 通过属性准确定位子节点
    result_nodes = get_node_by_keyvalue(nodes, {"name":"BProcesser"})
      #C. 修改节点属性
    change_node_properties(result_nodes, {"age": "1"})
      #D. 删除节点属性
    change_node_properties(result_nodes, {"value":""}, True)
    
    #3. 节点修改
      #A.新建节点
    a = create_node("person", {"age":"15","money":"200000"}, "this is the firest content")
      #B.插入到父节点之下
    add_child_node(result_nodes, a)
    
    #4. 删除节点
       #定位父节点
    del_parent_nodes = find_nodes(tree, "processers/services/service")
       #准确定位子节点并删除之
    target_del_node = del_node_by_tagkeyvalue(del_parent_nodes, "chain", {"sequency" : "chain1"})
    
    #5. 修改节点文本
       #定位节点
    text_nodes = get_node_by_keyvalue(find_nodes(tree, "processers/services/service/chain"), {"sequency":"chain3"})
    change_node_text(text_nodes, "new text")
    
    #6. 输出到结果文件
    write_xml(tree, "./out.xml")
    