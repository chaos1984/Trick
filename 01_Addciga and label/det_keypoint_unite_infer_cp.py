# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import cv2
import math
import numpy as np
import paddle
from hand import *

from det_keypoint_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from visualize import draw_pose
from benchmark_utils import PaddleInferBenchmark
from utils import get_current_memory_mb
from keypoint_postprocess import translate_to_ori_images
from predict_for_training import indent,get_annotations
import xmltodict

def labelsmoke(data):
    return

KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.det_times.report(average=True)
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }

    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, det_threshold,
                           keypoint_threshold, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res, det_threshold)
    keypoint_vector = []
    score_vector = []
    rect_vector = det_rects
    batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

    for i in range(batch_loop_cnt):
        start_index = i * keypoint_batch_size
        end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
        batch_images = rec_images[start_index:end_index]
        batch_records = np.array(records[start_index:end_index])
        if run_benchmark:
            # warmup
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=False)
            # run benchmark
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=True)
        else:
            keypoint_result = keypoint_detector.predict(batch_images,
                                                        keypoint_threshold)
        orgkeypoints, scores = translate_to_ori_images(keypoint_result,
                                                       batch_records)
        keypoint_vector.append(orgkeypoints)
        score_vector.append(scores)

    keypoint_res = {}
    keypoint_res['keypoint'] = [
        np.vstack(keypoint_vector).tolist(), np.vstack(score_vector).tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1,
                          save_res=False):
    # Hand_ciga
    cigafiledir = r"/data/wangyj/02_Study/PaddleDetection/addciga/ciga/"
    cigafiles = glob.glob(cigafiledir + '*.png')     
    #
    det_timer = detector.get_timer()
    store_res = []
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            # warmup
            results = detector.predict(
                [image], FLAGS.det_threshold, repeats=10, add_timer=False)
            # run benchmark
            results = detector.predict(
                [image], FLAGS.det_threshold, repeats=10, add_timer=True)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            results = detector.predict([image], FLAGS.det_threshold)
        if results['boxes_num'] == 0:
            continue
        person = []
        for i in results["boxes"]:
           
            if i[0]==0 and i[1]>0.5:
                # print (i)

                person.append(i.tolist())

        
        keypoint_res = predict_with_given_det(
            image, results, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.det_threshold, FLAGS.keypoint_threshold, FLAGS.run_benchmark)

        if save_res:
            store_res.append([
                i, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            draw_pose(
                img_file,
                keypoint_res,
                visual_thread=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)
        # Add ciga in img 
        print (img_file)
        img = cv2.imread(img_file)
        img_copy = img.copy()
        filename = img_file.split(os.sep)[-1].split('.')[-2]
        counter = 0  
        add_ciga_threshold = 0.8
        bndboxlist = []
        for i in range(len(keypoint_res['keypoint'][0])):
            res = keypoint_res['keypoint'][0][i][7:11]
            if res[2][2] > add_ciga_threshold :
                counter += 1
                cigafile = random.sample(cigafiles, 1)[0]
                # print (cigafile)
                ciga = cv2.imread(cigafile)
                img,bndbox = addciga(img,ciga,res[0],res[2])
                if bndbox !=[]:
                    bndboxlist.append(bndbox)
            if res[3][2] > add_ciga_threshold :
                counter += 1
                cigafile = random.sample(cigafiles, 1)[0]
                ciga = cv2.imread(cigafile)
                # print (cigafile)
                img,bndbox = addciga(img,ciga,res[1],res[3])
                if bndbox !=[]:
                    bndboxlist.append(bndbox)
        #
        xml_file = img_file.replace(".jpg",".xml")
        # print ("xml_file")
        object = getObjectxml(xml_file,'smoke')
        bndboxlist =[]
        # print ("object")
        # print (len(object))
        # print (object)
        for i in object:
            # print (i['bndbox']['xmin'])
            # print (int(i['xmin']),int(i['ymin']),int(i['ymax']),int(i['xmax']))
            # 
            bndboxlist.append([1,1,i['bndbox']['xmin'],i['bndbox']['ymin'],i['bndbox']['xmax'],i['bndbox']['ymax']])


        #
        results = {}
        person.extend(bndboxlist)
        
        results['boxes'] = person
        labels = ['person','smoke']
        labelxml = get_annotations(img, results, labels)
        if bndboxlist != []:

            save('/data/wangyj/02_Study/PaddleDetection/addciga/out',filename,img_copy,labelxml)
        else:
            print ("Failure:",img_file)
        # print (type(labelxml))
        # xmlfile = img_file.replace(".jpg",".xml")
        # labelxml.write(xmlfile, encoding='utf-8')
        # print ("bndbox")
        # print (bndbox)
        # saveimgfile = img_file.replace(".jpg","_0.jpg")
        # print ("Total ciga:",counter)
        # print (saveimgfile)
        # cv2.imwrite(saveimgfile,img)
        #        
    if save_res:
        """
        1) store_res: a list of image_data
        2) image_data: [imageid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_image_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def topdown_unite_predict_video(detector,
                                topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1,
                                save_res=False):
    # Hand_ciga
    cigafiledir = r"/data/wangyj/02_Study/PaddleDetection/addciga/ciga/"
    cigafiles = glob.glob(cigafiledir + '*.png')     
    #
    video_name = 'output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.splitext(os.path.basename(FLAGS.video_file))[
            0] + '.mp4'
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print (width,height)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 0
    store_res = []
    timecost = 0
    while (1):
        ret, frame = capture.read()
        frame_ori = frame
        if not ret:
            break
        index += 1

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.predict([frame2], FLAGS.det_threshold)
        
        start = time.perf_counter()
        keypoint_res = predict_with_given_det(
            frame2, results, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.det_threshold, FLAGS.keypoint_threshold, FLAGS.run_benchmark)
        timecost = (time.perf_counter() - start)
        print('detect frame: %d, time cost:%f s' % (index,timecost))
        
        im = draw_pose(
            frame,
            keypoint_res,
            visual_thread=FLAGS.keypoint_threshold,
            returnimg=True)
        if save_res:
            store_res.append([
                index, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])
        # Add ciga in img 
        counter = 0  
        add_ciga_threshold = 0.8
        for i in range(len(keypoint_res['keypoint'][0])):
            res = keypoint_res['keypoint'][0][i][7:11]
            if res[2][2] > add_ciga_threshold :
                counter += 1
                cigafile = random.sample(cigafiles, 1)[0]
                ciga = cv2.imread(cigafile)
                img = addciga(frame_ori,ciga,res[0],res[2])
            if res[3][2] > add_ciga_threshold :
                counter += 1
                cigafile = random.sample(cigafiles, 1)[0]
                ciga = cv2.imread(cigafile)
                img = addciga(frame_ori,ciga,res[1],res[3])
        print ("Total ciga:",counter)
        savaimgfile = r"/data/wangyj/02_Study/PaddleDetection/PaddleDetection/output/out.png"
        cv2.imwrite(savaimgfile,img)
        #        
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    writer.release()
    if save_res:
        """
        1) store_res: a list of frame_data
        2) frame_data: [frameid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_video_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def addciga(img,ciga,elbow,wrist,checkratio=0.2):
    '''
    Description: add ciga in hands
    Author: Yujin Wang
    Date: 2022-01-14
    Args:
        img[img]:BG image
        ciga[img]:ciga image
        elbow,wrist[list]:key-points of person
        checkratio[0~1 float]:only center area of bg will be checked.
    Return:
        img[img]: add ciga in hands figure
        bndbox[list]: ciga label box 
    Usage:
    '''
        h,w,_ = img.shape
        checkrange = [int(w*checkratio),int(w-w*checkratio),int(h*checkratio),int(h-h*checkratio)]
        # print ("checkrange")
        # print (checkrange)
        hand_x,hand_y,dis,angle = get_hand_point(elbow_x=elbow[0], elbow_y=elbow[1], wrist_x=wrist[0], wrist_y=wrist[1])
        pos = [int(hand_x),int(hand_y)]
        ratio = 0.1
        rotate_ciga = img_rotate(ciga,angle,scale=ratio)
        img,bndbox = paste_img(img,rotate_ciga,pos,1,checkrange)
        return img,bndbox

def getObjectxml(xmlfile,object):
    '''
    Description: Get dest object information
    Author: Yujin Wang
    Date: 2022-1-6
    Args: 
        xmlfile[str]: .xml file from labelimg
        objct[str]: Dest image
    Return:
        obj[list]: obeject list
    Usage:
    '''
    f = open(xmlfile)
    xmldict =  xmltodict.parse(f.read())
    obj = []
    try: 
        len(xmldict['annotation']["object"]) # Check "object" in keys
        try:
            # For multi-object
            for i in xmldict['annotation']["object"]:
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


def main():
    pred_config = PredictConfig(FLAGS.det_model_dir)
    detector_func = 'Detector'
    if pred_config.arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'
    
    detector = eval(detector_func)(pred_config,
                                   FLAGS.det_model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
    assert KEYPOINT_SUPPORT_MODELS[
        pred_config.
        arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'
    topdown_keypoint_detector = KeyPoint_Detector(
        pred_config,
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)
    # predict from video file or camera video stream
    
    if FLAGS.video_file is not None or FLAGS.camera_id != -1: # IMAGE IS RUNING IN THIS IF-ELSE
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                    FLAGS.save_res) 
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size, FLAGS.save_res)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
 
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')


if __name__ == '__main__':
    
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    
    main()
