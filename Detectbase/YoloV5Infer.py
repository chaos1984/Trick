from Detectbase import Infer
import logging
import argparse
import ntpath
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from threading import Thread


# from models.common import DetectMultiBackend
from models.experimental import attempt_load
# from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.datasets import create_dataloader
from utils.general import ( box_iou, check_dataset, check_img_size,
                           coco80_to_coco91_class,  non_max_suppression,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

LOGGER = logging.getLogger(__name__)

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

class ObjectDetect(Infer):
    def __init__(self,config):
        Infer.__init__(self, config)
        # self.imgsz = (640, 640)

        self.max_det=1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        # self.line_thickness = 3  # bounding box thickness (pixels)
        self.half = True
        # use FP16 half-precision inference
        self.dnn = False
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        try:
            self.model.model.half() if self.half else self.model.model.float()
        except:
            pass
    @torch.no_grad()
    def run(self, img,res={}):
        # img = cv2.imread(img)
        img0 = img.copy()
        img = letterbox(img, self.imgsize, stride=self.stride, auto=True)[0] #True
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        t1 = time_synchronized()
        pred = self.model(im, augment=False, visualize=False)
        t2 = time_synchronized()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=1000)
        t3 = time_synchronized()
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        # Process predictions

        self.res = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            self.res.append(det)

class Validation(Infer):
    def __init__(self,config):
        Infer.__init__(self, config)
        self.batch_size = 8  # batch size
        self.max_det=1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        # self.line_thickness = 3  # bounding box thickness (pixels)
        self.half = False  # use FP16 half-precision inference
        self.dnn = False
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.model.half() if self.half else self.model.model.float()


    @torch.no_grad()
    def run(self, rundir,save_dir):
        from utils.metrics import ConfusionMatrix, ap_per_class
        from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
        # from utils.callbacks import Callbacks
        # callbacks = Callbacks(),
        single_cls = False
        # Directories


        # Configure
        self.model.eval()

        nc = int(len(self.model.names))  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        testdir = rundir + '/test.txt'
        dataloader = create_dataloader(testdir, self.imgsize[0], self.batch_size, self.stride, single_cls, pad=0.0, rect=self.pt,
                                       workers=2)[0]
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc,conf=self.conf_thres, iou_thres=self.iou_thres)
        names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss = torch.zeros(3, device=self.device)
        jdict, stats, ap, ap_class = [], [], [], []
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            im = im.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            # Inference
            out, train_out = self.model(im, augment=False, val=True)  # inference, loss outputs
            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(self.device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            lb = []
            out = non_max_suppression(out, self.conf_thres, self.iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
                # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # Plot images
            if batch_i < 3:
                f = save_dir +f'/val_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
                f = save_dir + f'/val_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if  nc < 50 and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        # shape = (self.batch_size, 3, self.imgsize[0], self.imgsize[1])
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # callbacks.run('on_val_end')

        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps