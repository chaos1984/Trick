import os
import cv2 
import numpy as np
# from numpy.lib.function_base import angle
import glob
import random

# from rule import get_hand_point
from predict_for_training import indent,get_annotations

def save(output_dir,filename,img,labelxml):
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    labelxml.write(output_dir + os.sep + filename + '.xml', encoding='utf-8')
    cv2.imwrite(output_dir + os.sep + filename + '.jpg', img)
    

def getBndbox(img,scale=1):
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
    '''

    fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(fg_img_gray, 10, 255, cv2.THRESH_BINARY)
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

def img_rotate(img,angle,scale=1):
    '''
    Description: Rotate and scale image
    Author: Yujin Wang
    Date: 2022-01-11
    Args:
        img[img]:Source image
        angle[deg]:float
        scale[float]:Scale raitio
    Return:
    Usage:
    '''
    h,w,_ =img.shape
    matRotate = cv2.getRotationMatrix2D((0.5*w,0.5*h), angle+180, scale)
    
    return cv2.warpAffine(img, matRotate, (int(w),int(h))) 
    

def dist(x1, x2, y1, y2):
    """
    欧氏距离
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def ang(x1, x2, y1, y2):
    """
    角度
    """
    tan = (y1-y2)/(x1-x2)
    # print (np.arctan(tan))
    return np.arctan(tan)/np.pi*180

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
    # cv2.resizeWindow(name,640, 640)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_hand_point(elbow_x, elbow_y, wrist_x, wrist_y):
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

def paste_img(bg_img,fg_img,pos,flag_bg,checkrange):
    
    h,w,_ = fg_img.shape
    c_h = int(h/2);c_w = int(w/2)
    roi_h1 = pos[1]-c_h;roi_h2=pos[1]-c_h+h;roi_w1=pos[0]-c_w;roi_w2=pos[0]-c_w+w
    # print ("pos")
    # print (pos)
    # print (pos[0]>checkrange[0] , pos[0]<checkrange[1] , pos[1]>checkrange[2] , pos[1]>checkrange[3])
    # print (c_h,c_w)
    if flag_bg == 1:
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
                # print (pos[1]-c_h,pos[1]-c_h+h,pos[0]-c_w,pos[0]-c_w+w)
                return bg_img,[]
        else:
            return bg_img,[]
        
    # bg_img[pos[1]-c_h:pos[1]-c_h+h,pos[0]-c_w:pos[0]-c_w+w] = img2
    # bndbox = [int(roi_w1+bndbox[0]), int(roi_h1+bndbox[1]), int(roi_w1+bndbox[2]),int(roi_h1+bndbox[3])]
    # cv2.rectangle(bg_img,pt1=(int(roi_w1+bndbox[0]), int(roi_h1+bndbox[1])), pt2=(int(roi_w1+bndbox[2]), int(roi_h1+bndbox[3])),color=(255, 125, 125), thickness=1)
    # cv_show("contour",img)
    # bndbox = [int(roi_w1+bndbox[0]), int(roi_h1+bndbox[1]), int(roi_w1+bndbox[2]),int(roi_h1+bndbox[3])]
    # return bg_img,bndbox


if __name__ == "__main__":
    
    # imgfiledir =r"./img/"
    # imgfiles = glob.glob(imgfiledir + '*.png')

    cigafiledir = r"./ciga/"
    cigafiles = glob.glob(cigafiledir + '*.png')
    ciga = random.sample(cigafiles, 1)[0]


    # save_path = imgfiledir+"ciga"+'/'

    res =   [[905.21044921875, 435.4527587890625, 0.7316806316375732], [801.7477416992188, 478.2089538574219, 0.7653260827064514], [904.3194580078125, 457.283935546875, 0.5500354170799255], [862.6246337890625, 477.6883544921875, 0.7263373732566833]]
    left_hand_x,left_hand_y,left_dis,left_angle = get_hand_point(elbow_x=res[0][0], elbow_y=res[0][1], wrist_x=res[2][0], wrist_y=res[2][1])
    right_hand_x,right_hand_y,right_dis,right_angle = get_hand_point(elbow_x=res[1][0], elbow_y=res[1][1], wrist_x=res[3][0], wrist_y=res[3][1])
    pos = [int(left_hand_x),int(left_hand_y)]
    pos = [int(right_hand_x),int(right_hand_y)]
    ratio = left_dis/27*0.1 
    img = cv2.imread("./smoke10.jpg")
    print (ciga)
    ciga = cv2.imread(ciga)
    # h,w,_ =ciga.shape
    # cv_show("ciga",ciga)
    # size = (int(w*ratio), int(h*ratio))  
    # ciga = cv2.resize(ciga, size, interpolation=cv2.INTER_AREA)  
    rotate_ciga = img_rotate(ciga,left_angle,scale=ratio)
    # img = img_rotate(img)
    # cv_show("ro",rotate_ciga)
    # h,w,_ =rotate_ciga.shape
    # cv_show("ciga",ciga)
    
    # cv2.rectangle(img,(int(left_hand_x),int(left_hand_y)),(int(right_hand_x),int(right_hand_y)),(0,0,255),10)


    # rotate_ciga_trans = subsBG(rotate_ciga)
    img = paste_img(img,rotate_ciga,pos,1)
    # cv_show("cv",img)
    # saveimgfile = save_path+
    # cv2.imwrite(saveimgfile,img)
    
    # print ("left_hand",left_hand_x,left_hand_y)
    # print ("right_hand",right_hand_x,right_hand_y)
