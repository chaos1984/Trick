import os
import cv2 
import numpy as np
# from numpy.lib.function_base import angle
import glob
import random

# from rule import get_hand_point
from predict_for_training import indent,get_annotations

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
    '''
    height, width, _ = img.shape
    xmin = int(object['xmin']); ymin= int(object['ymin']); xmax = int(object['xmax']); ymax = int(object['ymax'])
    h = ymax - ymin; w = xmax - xmin
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
    cv2.putText(img, label, (xmax,ymax), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    # cv_show('img',img)
    # print (saveimgfile)
    return img

def getFrame(dir,flielist,save_path=r"./save_each_frames_front"):
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
    num = 0
    for index,file in enumerate(dir):
        num += 1
        cap = cv2.VideoCapture()
        print (file)
        cap.open(file)
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

            if frame_num % 230 == 0:
                img_path = save_path+ "//" +str(num)+'_'+str(frame_num)+'_'+flielist[index][:-4]+".jpg"
                print (img_path)
                cv2.imwrite(img_path,frame)
            frame_num = frame_num + 1
        
            # wait 10 ms and if get 'q' from keyboard  break the circle
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()

def LabelObjectBaseKeypoint(img_file,
                            personbox,
                            keypoint_res ,
                            start = 0,
                            end = 5,
                            ratio_w = 0.18,
                            ratio_h = 0.18
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

def rotimg(img,angle,scale=1):
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
    h,w,_ =img.shape
    matRotate = cv2.getRotationMatrix2D((0.5*w,0.5*h), angle+180, scale)
    
    return cv2.warpAffine(img, matRotate, (int(w),int(h))) 
   
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

def pasteimg(bg_img,fg_img,pos,checkrange):
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
        img = pasteimg(img,rotate_ciga,pos,1)
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

def saveCropImg(img,object,saveimgfile,scale=1):
    '''
    Description: Crop image accoding to the bounding box from xml, and save the cropped image
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        img[image]: image
        object: dest object
        saveimgfile: croped image is saved in dir and file name
    Return:
        NAN
    Usage:
    '''
    height, width, _ = img.shape
    xmin = int(object['xmin']); ymin= int(object['ymin']); xmax = int(object['xmax']); ymax = int(object['ymax'])
    h = ymax - ymin; w = xmax - xmin
    x1, y1 = max(int(xmin-scale*w/2),0), max(int(ymin-scale*h/2),0)
    x2, y2 = min(int(xmax+scale*w/2),width), min(int(ymin+scale*h/2),height)
    dst = img[y1:y2,x1:x2]
    cv2.imwrite(saveimgfile,dst)


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