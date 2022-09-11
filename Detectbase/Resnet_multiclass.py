import os
#import PIL
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F


class ResnetDetector():
    def __init__(self, config) -> None:
        self.device = 'cpu'
        self.num_classes = len(config['classes'])
        self.img_size = config['imgsize']
        self.model = models.resnet18(num_classes=self.num_classes)
        num_frs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_frs, self.num_classes)
        assert os.path.exists(config["weights"]),'No weight found!'

        checkpoint = torch.load(config["weights"],map_location=self.device)

        self.model = self.model.to(self.device)

        self.model.load_state_dict(checkpoint,strict=False)

        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def predict(self, im): ###直接预测单张图#
        try:
            im = Image.fromarray(im)
        except:
            im = Image.open(im)
        #im  =cv2.imread(im)
        im = self.transform(im)
        im = torch.unsqueeze(im, dim=0)
        with torch.no_grad():
            output = torch.squeeze(self.model(im)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        return predict_cla