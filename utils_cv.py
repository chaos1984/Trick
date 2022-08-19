from utils_pre import *
from utils_xml import *
from utils_math import *




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
            return cv2.flip(img,0)
        if fliptype == 'h':
            return cv2.flip(img,1)
        if fliptype == 'vh':
            return cv2.flip(img,-1)
    else:
        return img

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
        h_norm = (hue/np.max(hue)*180).astype(dtype)
        s_norm = (sat/np.max(sat)*255).astype(dtype)
        v_norm = (val/np.max(val)*255).astype(dtype)
        img_hsv = cv2.merge([h_norm,s_norm,v_norm])
        return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

def main_augmentImgs(imgdir,prob=[1,1,0,0]):
    '''
       Augmentation for Images
    '''
    imgfilespath,imgfiles= getFiles(imgdir,ImgType)
    fliptype = input("Flip type(v,h,vh):")
    savedir = mkFolder(imgdir,"augmentation_"+fliptype)

    for id,img in enumerate(imgfiles):
        im = cv2.imread(imgfilespath[id])
        files = findRelativeFiles(imgfilespath[id])


        im = reflectimg(im,prob[0],fliptype=fliptype)
        im = hsvadjust1(im,prob[1])
        im = rotimg(im, 30, prob[2],scale=1)
        im = rgb2gray(im,prob[3])
        imgdir = savedir / f"{img[:-4]}_aug_{fliptype}.jpg"
        xmldir = savedir / f"{img[:-4]}_aug_{fliptype}.xml"
        for file in files:
            if ".xml" in file:
                flipObjextxml(file,str(xmldir),fliptype)

        cv2.imwrite(imgdir.__str__(), im)

if __name__ == "__main__":
    
    try:
        action = sys.argv[1]
        file_dir = sys.argv[2]
        if file_dir[-1] != '/':
            file_dir = file_dir+os.sep
    except:
        action = ""
 
        file_dir = r"D:/01_Project/02_Baosteel/01_Input/Dataset/V4_20220801/maindefectforlabel/labelwork/done/test/"
        # pass
    try:
        if action == "augmentation":#augmentation
            print(main_augmentImgs.__doc__)
            main_augmentImgs(file_dir)
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    os.system("pause")

