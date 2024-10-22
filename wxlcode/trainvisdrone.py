import os
import sys
from ultralytics.models.yolo.model import YOLO


# 设置工作目录
os.getcwd()
sys.path.insert(0, os.getcwd())

# Load a pretrained model
model = YOLO("/workspace/ultralytics/cfg/models/11/yolo11.yaml")

# Train the model on your custom dataset
model.train(data="/workspace/ultralytics/cfg/datasets/ours_datasets/visdrone.yaml", epochs=100)

