import sys
from utils_pre import *
from utils_xml import *
from utils_math import *
import math
import cv2
import numpy as np
import random
import traceback

def voc2yolocoor(data,h,w):
    for id,a in enumerate(data):
        data[id] =[a[0],float(a[1]/w),float(a[2]/h),float(a[3]/w),float(a[4]/h),a[5]]
    return data


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def rotate90(im,
                       targets=(),
                      ):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0]
    width = im.shape[1]

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # # # Perspective
    # P = np.eye(3)
    # P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    # P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation
    R = np.eye(3)
    # a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    #random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=90, center=(0, 0), scale=1)

    # # Shear
    # S = np.eye(3)
    # S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    # S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = 0.5 * width  # x translation (pixels)
    T[1, 2] = 0.5 * height  # y translation (pixels)

    # Combined rotation matrix
    # M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    M = T @ R @ C  # order of operations (right to left) is IMPORTANT
    im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
    # if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
    #     if perspective:
    #         im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
    #     else:  # affine
    #         im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    # cv_show("rot",im)
    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # targets = voc2yolocoor(targets,height,width)
        targets = np.array(targets)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:,:2].reshape(n,4,2)
        for id in range(n):
            minx = int(min(xy[id][:,0]));miny = int(min(xy[id][:,1]))
            maxx = int(max(xy[id][:, 0]));maxy = int(max(xy[id][:, 1]))
            targets[id, :] = [minx,miny,maxx,maxy,targets[id,5],targets[id,0]]
              # xy = (xy[:, :2] /  xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        #
        # # create new boxes
        # x = xy[:, [0, 2, 4, 6]]
        # y = xy[:, [1, 3, 5, 7]]
        # new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        # new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        # new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        # i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # i = box_candidates(box1=targets[:, 1:5].T , box2=new.T, area_thr=0.01 )
        # targets = targets[i]

    return im, targets


def plotRectBox(img,object,label):
    '''
    Description: Plot bndbox and label in image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
        label[str]:label
    Return:
        img
    Usage:
        from utils_pre import getObjectxml
        xmlfile = 'xxx.xml';label='person'; saveimgfile='xxx.jpg'
        objectlist = getObjectxml(xmlfile,label)
        imgfile = xmlfile.replace(".xml",".tif")
        img = cv2.imread(imgfile)
        if len(objectlist) > 0:
            for index,object in enumerate(objectlist):
                imgname = "_%i.png" %(index)
                saveimgfile = xmlfile.replace(".xml",imgname)
                # saveCropImg(img,object['bndbox'],save_path+saveimgfile,scale=3)
                img = plotRectBox(img,object['bndbox'],label,saveimgfile)
            cv2.imwrite(saveimgfile,img)
        else:
            print ('Warnning: No %s found!' %(label))
    '''
    height, width, _ = img.shape
    xmin = int(object['xmin']); ymin= int(object['ymin']); xmax = int(object['xmax']); ymax = int(object['ymax'])
    h = ymax - ymin; w = xmax - xmin
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
    cv2.putText(img, label, (xmax,ymax), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    return img



def LabelObjectBaseKeypoint(img_file,
                            personbox,
                            keypoint_res ,
                            start = 0,
                            end = 5,
                            ratio_w = 0.18,
                            ratio_h = 0.18,
                            outdir ='/data/wangyj/02_Study/PaddleDetection/facemask/out',   
):
    '''
    Description: Label and box from keypoints
        COCO keypoint indexes:
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
    Author: Yujin Wang
    Date: 2022-01-20
    Args:
        img_file[img]: img file
        personbox[]: results from object detection
        keypoint_res[]: results from pos detetion
        start[int]: COCO keypoint indexes start index
        end[int]: COCO keypoint indexes end index
        outdir[str]: output directory.
        ratio_w，ratio_h[float]: object box scale ratio factor between person box
    Return:
        NaN
    Usage:
        LabelObjectBaseKeypoint(img_file,results["boxes"],keypoint_res,ratio_w=0.18,ratio_h=0.13，outdir='/data/wangyj/02_Study/PaddleDetection/facemask/out')
    '''
    # ratio_w = 0.18
    # ratio_h = 0.3
    person = []
    for i in personbox:
        if i[0]==0 and i[1]>0.5:
            # print (i)
            person.append(i.tolist())
    bndboxlist = []
    for i in range(len(keypoint_res['keypoint'][0])):
        res = keypoint_res['keypoint'][0][i][start:end]
        x = [i[0] for i in res];y = [i[1] for i in res]
        x_m = sum(x)/len(x);y_m = sum(y)/len(y)
        # print (person[i])
        object_w = int(abs(person[i][4]-person[i][2])*ratio_w);object_h = int(abs(person[i][5]-person[i][3])*ratio_h)
        bndboxlist.append([1,1,x_m-object_w,y_m-object_h,x_m+object_w,y_m+object_h])
    results = {}
    # print(person)
    person.extend(bndboxlist)
    results['boxes'] = person
    labels = ['person','mask']
    img = cv2.imread(img_file)
    filename = img_file.split(os.sep)[-1].split('.')[-2]
    labelxml = get_annotations(img, results, labels)
    if bndboxlist != []:
        save(outdir,filename,img,labelxml)
    else:
        print ("Failure:",img_file)

def savexmlimg(output_dir,filename,img,labelxml):
    '''
    Description: Save for xml and jpg for labelimg
    Author: Yujin Wang
    Date: 2022-01-20
    Args:
        output_dir[str]:Output dir for xml and img
        filename[str]:file name without format
        img[img]: img file
        labelxml[xml]:xml.tree
    Return:
    Usage:
        savexmlimg(outdir,filename,img,labelxml)
    '''
    try:
        os.mkdir(output_dir)
    except:
        pass
    labelxml.write(output_dir + os.sep + filename + '.xml', encoding='utf-8')
    cv2.imwrite(output_dir + os.sep + filename + '.jpg', img)
    

def getBndbox(img,scale=1):
    '''
    Description: Get rectangle from contour
    Author: Yujin Wang
    Date: 2022-01-19
    Args:
        img:
        scale[float]:Scale bndbox
    Return:
        Boundbox[list]:[x1,y1,x2,y2]
    Usage:
        fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        _ , mask = cv2.threshold(fg_img_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bndbox = getBndbox(mask_inv)

    '''
    c, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(c[1])
    
    # cv2.rectangle(img,pt1=(int(x-(scale-1)*0.5*w), int(y-(scale-1)*0.5*h)), pt2=(int(x+scale*w), int(y+scale*h)),color=(125, 125, 125), thickness=1)
    # cv_show("contour",img)
    x1,y1,x2,y2 = int(x-(scale-1)*0.5*w), int(y-(scale-1)*0.5*h),int(x+scale*w), int(y+scale*h)
    return [x1,y1,x2,y2]
    

def subsBG(bg_img,fg_img):
    '''
    Description: Replace the background of fg_img with bg_img
    Author: Yujin Wang
    Date: 2022-01-11
    Args:
        bg_img[img]
        fg_img[img]
    Return:
        dst[img]
    Usage:
        roi = bg_img[pos[1]-c_h:pos[1]-c_h+h,pos[0]-c_w:pos[0]-c_w+w]  
        img2,bndbox = subsBG(roi,fg_img)
    '''

    fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
    _ , mask = cv2.threshold(fg_img_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bndbox = getBndbox(mask_inv)
    # cv_show("mask",mask)
    # print (bg_img.shape)
    # print (fg_img.shape)
    if bg_img.shape == fg_img.shape:
        bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
        # cv_show("bg",bg)
        
        fg = cv2.bitwise_and(fg_img,fg_img, mask=mask)
        # cv_show("img_fg",fg)
        dst = cv2.add(fg, bg)
        return dst,bndbox
    else:
        return bg_img,[]

def rotimg(img,angle,prob=1.0,scale=1):
    '''
    Description: Rotate and scale image
    Author: Yujin Wang
    Date: 2022-01-11
    Args:
        img[img]:Source image
        angle[deg]:float
        scale[float]:Scale raitio
    Return:
        cv2.warpAffine(img, matRotate, (int(w),int(h))) 
    Usage:
        ciga = cv2.imread(ciga)
        rotate_ciga = rotimg(ciga,left_angle,scale=ratio)
    '''
    rd = random.random()
    if rd< prob:
        h,w,_ =img.shape
        matRotate = cv2.getRotationMatrix2D((0.5*w,0.5*h), (random.random()-0.5)*angle, 1.3)
    
        return cv2.warpAffine(img, matRotate, (int(w),int(h)))
    else:
        return img
   
def cv_show(name, img):
    '''
    Summary: image show
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        name[str]:window label
        img[np.arry]:img array
    Return:
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_hand_point(elbow_x, elbow_y, wrist_x, wrist_y):
    '''
    Description: Get hand location from keypoints
    Author: Yujin Wang
    Date: 2022-01-20
    Args:
        elbow_x, elbow_y, wrist_x, wrist_y[float]:keyporints[7:11]
    Return:
        hand_x, hand_y[float]:hand location
        dis[float]: hand and elbow
        angle[float]: hand and elbow
    Usage:
        get_hand_point(elbow_x=elbow[0], elbow_y=elbow[1], wrist_x=wrist[0], wrist_y=wrist[1])
    '''
    if abs(wrist_x - elbow_x) > 1e-6:
        k = (wrist_y - elbow_y) / (wrist_x - elbow_x)
        b = elbow_y - k * elbow_x

        delta_x = abs(elbow_x - wrist_x) * 0.4
        if wrist_x > elbow_x:
            hand_x = wrist_x + delta_x
            hand_y = k * hand_x + b
        else:
            hand_x = wrist_x - delta_x
            hand_y = k * hand_x + b
    else:
        delta_y = abs(elbow_y - wrist_y) * 0.4
        hand_x = wrist_x
        if wrist_y > elbow_y:
            hand_y = wrist_y + delta_y
        else:
            hand_y = wrist_y - delta_y
    dis = dist(hand_x, elbow_x, hand_y, elbow_y)
    angle = ang(hand_x, elbow_x, hand_y, elbow_y)
    return hand_x, hand_y, dis, angle

def pasteImg(bg_img,fg_img,pos,checkrange):
    '''
    Description: Past fb image on bg image
    Author: Yujin Wang
    Date: 2022-01-20
    Args:
        bg_img[img]:
        fg_img[img]:
        pos[list]:
        checktange[list]:[int(w*checkratio),int(w-w*checkratio),int(h*checkratio),int(h-h*checkratio)] used to check pos in img.w and img.h
    Return:
        bg_img[img]: pasted bg img with fg img
        bndbox[list]:[1,1,int(roi_w1+bndbox[0]), int(roi_h1+bndbox[1]), int(roi_w1+bndbox[2]),int(roi_h1+bndbox[3])] roi boundbox list
    Usage:
        h,w,_ = img.shape
        checkrange = [int(w*checkratio),int(w-w*checkratio),int(h*checkratio),int(h-h*checkratio)]
        img = pasteImg(img,rotate_ciga,pos,1)
    '''
    h,w,_ = fg_img.shape
    c_h = int(h/2);c_w = int(w/2)
    roi_h1 = pos[1]-c_h;roi_h2=pos[1]-c_h+h;roi_w1=pos[0]-c_w;roi_w2=pos[0]-c_w+w
    if  pos[0]>checkrange[0] and pos[0]<checkrange[1] and pos[1]>checkrange[2]:
        # print ("Find pos")
        if min(pos[1]-c_h,pos[1]-c_h+h,pos[0]-c_w,pos[0]-c_w+w) > 0:
            roi = bg_img[pos[1]-c_h:pos[1]-c_h+h,pos[0]-c_w:pos[0]-c_w+w]  
            img2,bndbox = subsBG(roi,fg_img)
            bg_img[pos[1]-c_h:pos[1]-c_h+h,pos[0]-c_w:pos[0]-c_w+w] = img2
            if bndbox !=[]:
                bndbox = [1,1,int(roi_w1+bndbox[0]), int(roi_h1+bndbox[1]), int(roi_w1+bndbox[2]),int(roi_h1+bndbox[3])]
                return bg_img,bndbox
            else:
                return bg_img,[]
        else:
            return bg_img,[]


def reflectimg(img,prob=1.0,fliptype = 'v'):
    if random.random() < prob:
        if fliptype == 'v':
            return cv2.flip(img,0),True
        if fliptype == 'h':
            return cv2.flip(img,1),True
        if fliptype == 'vh':
            return cv2.flip(img,-1),True
    else:
        return img,False

def rgb2gray(img,prob=1.0):
    if random.random() < prob:
        return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        return img

def hsvadjust1(img,prob=1.0):
    '''
    Description:
        Change img HSV with maxmin
    Author: Yujin Wang
    Date: 2022-02-24
    Args:
        img[]
    Return:
    Usage:
    '''
    # print(adjh,adjs,adjv)
    if random.random() < prob:
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8
        if np.max(hue)!=0:
            h_norm = (hue/np.max(hue)*180).astype(dtype)
            s_norm = (sat/np.max(sat)*255).astype(dtype)
        else:
            h_norm = hue.astype(dtype)
            s_norm = sat.astype(dtype)
        v_norm = (val/np.max(val)*255).astype(dtype)
        img_hsv = cv2.merge([h_norm,s_norm,v_norm])
        return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    else:
        return img

def main_augmentImgs(imgdir,prob=[1,0.5,0,0]):
    '''
       Augmentation for Images
    '''
    imgfilespath,imgfiles= getFiles(imgdir,ImgType)
    fliptype = input("Flip type(v,h,vh):")
    savedir = mkFolder(imgdir,"augmentation_"+fliptype)

    for id,img in enumerate(imgfiles):
        im = cv2.imread(imgfilespath[id])
        files = findRelativeFiles(imgfilespath[id])


        im,_= reflectimg(im,prob[0],fliptype=fliptype)
        im = hsvadjust1(im,prob[1])
        im = rotimg(im, 30, prob[2],scale=1)
        im = rgb2gray(im,prob[3])
        imgdir = savedir / f"{img[:-4]}_aug_{fliptype}.jpg"
        xmldir = savedir / f"{img[:-4]}_aug_{fliptype}.xml"
        for file in files:
            if ".xml" in file:
                flipObjextxml(file,str(xmldir),fliptype)
        cv2.imwrite(imgdir.__str__(), im)

def main_random_perspective(imgdir):
    '''
       Augmentation for Images
    '''
    imgfilespath,imgfiles= getFiles(imgdir,ImgType)
    savedir = mkFolder(imgdir, "augmentation")
    for id,imgfile in enumerate(imgfilespath):
        print("here")
        im = cv2.imread(imgfile)
        files = findRelativeFiles(imgfile)
        xmldir = imgdir + f"{imgfiles[id][:-4]}.xml"
        for file in files:
            if ".xml" in file:
                objectlist,w,h = getObjectxml(xmldir,classes='all')

        img, label = rotate90(im,
                               targets=objectlist,
   )
    return 0



if __name__ == "__main__":
    
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
 
        file_dir = r"D:\test/"
        # pass
    try:
        if action == "augmentation":#augmentation
            print(main_augmentImgs.__doc__)
            main_augmentImgs(file_dir)
        if action ==  "":
            main_random_perspective(file_dir)
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    os.system("pause")

