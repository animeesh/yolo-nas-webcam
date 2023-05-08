import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models

model = models.get(Models.YOLOX_N,pretrained_weights= "coco")
model.predict_webcam()
