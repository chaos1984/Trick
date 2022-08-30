import sys
import torch

class Infer():
    def __init__(self,config):
        self.weights = config["weights"]
        # self.classes = config["class"]
        self.data = ''
        # self.save_dir = config["save_dir"]
        self.conf_thres = config["conf_thres"] # confidence threshold
        self.iou_thres =  config["iou_thres"]  # NMS IOU threshold
        self.imgsize = (config["imgsize"],config["imgsize"]) #imgsize
        if torch.cuda.is_available() and config["device"] == 'GPU':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(config["device"])

