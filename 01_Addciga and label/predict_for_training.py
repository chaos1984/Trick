import os
import yaml
import glob
from functools import reduce

import cv2
import numpy as np
import math
import time
import paddle
import argparse
import ast
from visualize import visualize_box_mask
from infer import Detector
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint, KEYPOINT_SUPPORT_MODELS
from det_keypoint_unite_infer import predict_with_given_det
from preprocess import decode_image
from keypoint_preprocess import expand_crop
from infer import Detector, PredictConfig, print_arguments, get_test_images
from PIL import Image, ImageDraw, ImageFont
from visualize import get_color_map_list, get_color
import matplotlib.pyplot as plt
import matplotlib

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree


def dist2(x1, x2, y1, y2):
    """
    欧氏距离（的平方）
    """
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def make_bigger_box(xmin, ymin, xmax, ymax, scale=0.2):
    """
    将box (xmin, ymin, xmax, ymax) 按比例scale进行扩大:
    """
    xmin = xmin * (1 - scale)
    ymin = ymin * (1 - scale)
    xmax = xmax * (1 + scale)
    ymax = ymax * (1 + scale)
    return xmin, ymin, xmax, ymax


def point_in_box(x, y, xmin, ymin, xmax, ymax):
    """
    判断点(x,y)是否在box (xmin, ymin, xmax, ymax)之中
    """
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def box_in_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """
    判断box (x1, y1, x2, y2)是否在box (xmin, ymin, xmax, ymax) 之中
    """
    return (xmin <= x1 <= xmax) and (xmin <= x2 <= xmax) and (ymin <= y1 <= ymax) and (ymin <= y2 <= ymax)


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--get_training_data_from_video",
        type=str,
        default=False,
        required=False)
    parser.add_argument(
        "--det_model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--keypoint_model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=False)
    parser.add_argument(
        "--keypoint_batch_size",
        type=int,
        default=1,
        help=("batch_size for keypoint inference. In detection-keypoint unit"
              "inference, the batch size in detection is 1. Then collate det "
              "result in batch for keypoint inference."))
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--det_person_threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--det_smoke_threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--det_phone_threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='fluid',
        help="mode of running(fluid/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
             "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        '--use_dark',
        type=bool,
        default=True,
        help='whether to use darkpose to get better keypoint position predict ')

    return parser


def draw_box_keypoint(
        im,
        det_results,
        labels,
        keypoint_results,
        keypoint_threshold):
    # det results
    np_boxes = det_results['boxes']

    im = Image.fromarray(im)
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                  'right_bottom:[{:.2f},{:.2f}]'.format(
                int(clsid), score, xmin, ymin, xmax, ymax))
            # draw bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=draw_thickness,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        # draw label
        fontsize = 20
        font = ImageFont.truetype(font='/usr/share/fonts/chinese/msyh.ttc', size=fontsize)
        label_cn = labels[clsid]
        if labels[clsid] == 'smoke':
            label_cn = '香烟'
        elif labels[clsid] == 'person':
            label_cn = '人'
        elif labels[clsid] == 'cell phone':
            label_cn = '手机'

        text = "{} {:.4f}".format(label_cn, score)
        tw, th = draw.textsize(text, font=font)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255), font=font)
    im = np.array(im)

    if keypoint_results is None:
        return im

    # keypoint results
    skeletons, scores = keypoint_results['keypoint']
    kpt_nums = 17
    if len(skeletons) > 0:
        kpt_nums = skeletons.shape[1]
    if kpt_nums == 17:  # plot coco keypoint
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    else:  # plot mpii keypoint
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                 (8, 13)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    color_set = keypoint_results['colors'] if 'colors' in keypoint_results else None

    canvas = im.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < keypoint_threshold:
                continue
            color = colors[i] if color_set is None else colors[color_set[j] % len(colors)]

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                color,
                thickness=-1)

    to_plot = cv2.addWeighted(im, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] < keypoint_threshold or skeletons[j][edge[1], 2] < keypoint_threshold:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            color = colors[i] if color_set is None else colors[color_set[j] % len(colors)]
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


def create_element_node(name, text='', tail='\n'):
    node = ET.Element(name)
    node.text = text
    node.tail = tail
    return node


def indent(elem, level=0):
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


def get_annotations(im, results, labels):
    root = create_element_node('annotation')

    root.append(create_element_node('folder'))
    root.append(create_element_node('filename'))
    root.append(create_element_node('path'))

    source = create_element_node('source')
    source.append(create_element_node('database', 'ZhangZhe'))
    root.append(source)

    size = create_element_node('size')
    h, w, c = im.shape
    size.append(create_element_node('width', str(w)))
    size.append(create_element_node('height', str(h)))
    size.append(create_element_node('depth', str(c)))
    root.append(size)

    root.append(create_element_node('segmented', '0'))

    np_boxes = results['boxes']
    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox

            # 创建新的box
            new_obj = create_element_node('object')

            bndbox = create_element_node('bndbox')
            bndbox.append(create_element_node('xmin', str(int(xmin))))
            bndbox.append(create_element_node('ymin', str(int(ymin))))
            bndbox.append(create_element_node('xmax', str(int(xmax))))
            bndbox.append(create_element_node('ymax', str(int(ymax))))

            new_obj.append(create_element_node('name', labels[clsid]))
            new_obj.append(create_element_node('pose', 'Unspecified'))
            new_obj.append(create_element_node('truncated', '0'))
            new_obj.append(create_element_node('difficult', '0'))
            new_obj.append(bndbox)

            root.append(new_obj)
    indent(root)
    return ElementTree(element=root)


def get_person_from_box(im, results):
    box_image_list = []
    for box in results['boxes']:
        box_image, new_box, org_box = expand_crop(im, box, expand_ratio=0.2)
        if box_image is not None and box_image.size != 0:
            box_image_list.append(box_image)
    return box_image_list


def predict_images(detector, topdown_keypoint_detector, img_list,
                         person_detection_threshold, smoke_detection_threshold, phone_detection_threshold,
                         keypoint_detection_threshold, output_dir):
    labels = detector.pred_config.labels
    keypoint_dir = output_dir + os.sep + 'keypoint'
    if not os.path.exists(keypoint_dir):
        os.mkdir(keypoint_dir)

    for i, img_file in enumerate(img_list):
        print(img_file)
        filename = img_file.split(os.sep)[-1].split('.')[-2]
        # image, _ = decode_image(img_file, {})
        image = cv2.imread(img_file)
        results = detector.predict([img_file], 0.5)
        print(results)
        keypoint_result = None

        if results['boxes_num'] == 0:
            continue

        # 过滤
        abnormal_result = {'boxes': []}
        person_result = {'boxes': []}
        smoke_result = {'boxes': []}
        phone_result = {'boxes': []}
        for box in results['boxes']:
            clsid, bbox, score = int(box[0]), box[2:], box[1]
            if labels[clsid] == 'person' and score >= person_detection_threshold:
                person_result['boxes'].append(box)
                abnormal_result['boxes'].append(box)
            if labels[clsid] == 'smoke' and score >= smoke_detection_threshold:
                smoke_result['boxes'].append(box)
                abnormal_result['boxes'].append(box)
            if labels[clsid] == 'cell phone' and score >= phone_detection_threshold:
                phone_result['boxes'].append(box)
                abnormal_result['boxes'].append(box)
        print('abnormal_result: ', len(abnormal_result['boxes']))

        tree = get_annotations(image, abnormal_result, labels)

        if len(person_result['boxes']) != 0 and (len(smoke_result['boxes']) != 0 or len(phone_result['boxes']) != 0):
            doubtful_person_result = {'boxes': []}
            doubtful_phone_result = {'boxes': []}
            for person_box in person_result['boxes']:
                xmin, ymin, xmax, ymax = person_box[2:]
                xmin, ymin, xmax, ymax = make_bigger_box(xmin, ymin, xmax, ymax, scale=0.2)

                for phone_box in phone_result['boxes']:
                    x1, y1, x2, y2 = phone_box[2:]
                    if box_in_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
                        doubtful_phone_result['boxes'].append(phone_box)
                        doubtful_person_result['boxes'].append(person_box)

            doubtful_person_result['boxes'] = np.array(doubtful_person_result['boxes'])
            doubtful_phone_result['boxes'] = np.array(doubtful_phone_result['boxes'])

            if len(doubtful_person_result['boxes']) != 0:
                keypoint_result = predict_with_given_det(
                    image,
                    det_res=doubtful_person_result,
                    keypoint_detector=topdown_keypoint_detector,
                    keypoint_batch_size=1,
                    det_threshold=person_detection_threshold,
                    keypoint_threshold=keypoint_detection_threshold,
                    run_benchmark=False)

                skeletons, scores = keypoint_result['keypoint']
                box_images = get_person_from_box(image, doubtful_person_result)

                # 姿态输出
                for j, box_image in enumerate(box_images):
                    with open(keypoint_dir + os.sep + filename + '_keypoint_' + str(j) + '.txt', 'w+') as f:
                        for k in range(len(skeletons[j])):
                            s = ''
                            for q in range(len(skeletons[j][k])):
                                if q == 0:
                                    s += '%.6f' % skeletons[j][k][q]
                                else:
                                    s += ' %.6f' % skeletons[j][k][q]
                            s += '\n'
                            f.write(s)

                    cv2.imwrite(keypoint_dir + os.sep + filename + '_keypoint_' + str(j) + '.png', box_image)

        # 保存结果
        im = np.array(draw_box_keypoint(image, abnormal_result, labels, keypoint_result,
                                        keypoint_detection_threshold))
        cv2.imwrite(output_dir + os.sep + filename + '.jpg', im)
        tree.write(output_dir + os.sep + filename + '.xml', encoding='utf-8')


def predict_video(detector, topdown_keypoint_detector, video_path,
                  person_detection_threshold, keypoint_detection_threshold, output_dir):
    if video_path != -1:
        capture = cv2.VideoCapture(video_path)
        video_name = os.path.split(video_path)[-1]
    else:
        capture = cv2.VideoCapture(video_path)
        video_name = os.path.split(video_path)[-1]

    fps = 25
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count', frame_count)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # yapf: disable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # yapf: enable
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, video_name)
    print(out_path)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    index = 1
    labels = detector.pred_config.labels
    while (1):
        ret, frame = capture.read()
        if not ret:
            break

        print('detect frame:%d' % (index))
        index += 1

        results = detector.predict([frame], 0.5)

        # 过滤
        person_result = {'boxes': []}
        for box in results['boxes']:
            clsid, bbox, score = int(box[0]), box[2:], box[1]
            if labels[clsid] == 'person' and score >= person_detection_threshold:
                person_result['boxes'].append(box)
        person_result['boxes'] = np.array(person_result['boxes'])

        if len(person_result['boxes']) != 0:
            keypoint_result = predict_with_given_det(
                frame,
                det_res=person_result,
                keypoint_detector=topdown_keypoint_detector,
                keypoint_batch_size=1,
                det_threshold=person_detection_threshold,
                keypoint_threshold=keypoint_detection_threshold,
                run_benchmark=False)

            im = np.array(draw_box_keypoint(frame, person_result, labels, keypoint_result,
                                            keypoint_detection_threshold))

            im = np.array(im)
        else:
            im = np.array(frame)
        writer.write(im)
    writer.release()


def predict_video_for_images(detector, video_path, person_detection_threshold, output_dir):
    if video_path != -1:
        capture = cv2.VideoCapture(video_path)
        video_name = os.path.split(video_path)[-1]
    else:
        capture = cv2.VideoCapture(video_path)
        video_name = os.path.split(video_path)[-1]
    # yapf: disable
    # yapf: enable
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    index = 1
    labels = detector.pred_config.labels
    while (1):
        ret, frame = capture.read()
        if not ret:
            break

        print('detect frame:%d' % (index))
        index += 1
        # if index % 8 != 0:
        #     continue

        results = detector.predict([frame], 0.5)

        # 过滤
        person_result = {'boxes': []}
        for box in results['boxes']:
            clsid, bbox, score = int(box[0]), box[2:], box[1]
            if labels[clsid] == 'person' and score >= person_detection_threshold:
                person_result['boxes'].append(box)
        person_result['boxes'] = np.array(person_result['boxes'])
        if len(person_result['boxes']) > 0:
            tree = get_annotations(frame, person_result, labels)
            print(output_dir + os.sep + video_name.split('.')[0] + '_' + str(index) + '.jpg')
            cv2.imwrite(output_dir + os.sep + video_name.split('.')[0] + '_' + str(index) + '.jpg', frame)
            tree.write(output_dir + os.sep + video_name.split('.')[0] + '_' + str(index) +  '.xml', encoding='utf-8')


def main():
    pred_config = PredictConfig(FLAGS.det_model_dir)
    detector = Detector(
        pred_config,
        FLAGS.det_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    if FLAGS.keypoint_model_dir is not None:
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

    if FLAGS.video_file is None:
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        print(img_list)
        predict_images(detector, topdown_keypoint_detector, img_list,
                             FLAGS.det_person_threshold, FLAGS.det_smoke_threshold, FLAGS.det_phone_threshold,
                             FLAGS.keypoint_threshold, FLAGS.output_dir)
        detector.det_times.info(average=True)
        topdown_keypoint_detector.det_times.info(average=True)
    else:
        assert FLAGS.video_file is not None
        if FLAGS.get_training_data_from_video:
            predict_video_for_images(detector, FLAGS.video_file,
                                     FLAGS.det_person_threshold, FLAGS.output_dir)
        else:
            predict_video(detector, topdown_keypoint_detector, FLAGS.video_file,
                          FLAGS.det_person_threshold, FLAGS.keypoint_threshold, FLAGS.output_dir)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()

    main()
