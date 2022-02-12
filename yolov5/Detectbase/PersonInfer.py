from Detectbase import Infer
import argparse
import ntpath
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

class PersonInfer(Infer):
    def __init__(self,config):
        Infer.__init__(self, config)
        self.imgsz = (640, 640)
        self.max_det=1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        # self.line_thickness = 3  # bounding box thickness (pixels)
        self.half = False  # use FP16 half-precision inference
        self.dnn = False
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.model.model.half() if self.half else self.model.model.float()
    @torch.no_grad()
    def run(self, img,res={}):
        # img = cv2.imread(img)
        img0 = img.copy()
        img = letterbox(img, [640, 640], stride=self.stride, auto=True)[0]
        # cv2.imwrite('../dataset/test/11.jpg',img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # cv2.imwrite('../dataset/test/12.jpg', img)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=1000)

        # Process predictions
        self.res = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            self.res.append(det)