import sys
import torch

class Infer():
    def __init__(self,config):
        self.weights = config["weights"]
        self.data = config["data"]
        # self.save_dir = config["save_dir"]
        self.conf_thres = config["conf_thres"] # confidence threshold
        self.iou_thres =  config["iou_thres"]  # NMS IOU threshold
        self.device = torch.device(config["device"])