import time
import torch

import cv2
import matplotlib.pyplot as plt
from Detectbase import Infer
import torch
from mmpose.apis.inference import init_pose_model, inference_top_down_pose_model
import numpy as np

def yolo2bbox(data):
    res = []
    for i in data:
        i = i.tolist()
        if i[-1] == 0.0:
            res.append({'bbox': np.array(i[:4])})
    return res

class PoseInfer(Infer):
    def __init__(self,config):
        Infer.__init__(self, config)
        self.model = init_pose_model(self.data, self.weights, device=self.device)

    def run(self,img,bbox=[{'bbox': np.array([1014, 115, 1395, 990])}]):
        bbox = yolo2bbox(bbox['person'][0])
        results, _ = inference_top_down_pose_model(self.model, img, bbox,format='xyxy', dataset='TopDownCocoDataset')
        result = results[0]
        self.res = []
        for v in result['keypoints']:
            self.res.append(v)