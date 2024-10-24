import os
import sys
os.getcwd()
sys.path.insert(0, os.getcwd())

from ultralytics.models.yolo.model import YOLO

# Load a pretrained model
model = YOLO("/workspace/ultralytics/cfg/models/11/yolo11.yaml")

# Train the model on your custom dataset
model.train(data="/workspace/ultralytics/cfg/datasets/coco128.yaml", epochs=100)