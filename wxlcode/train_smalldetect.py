


import os
import sys
os.getcwd()
sys.path.insert(0, os.getcwd())

from ultralytics.models.yolo.model import YOLO
# from torch.utils.tensorboard import SummaryWriter

# log_dir = "/workspace/runs"
# tensorboard_callback = SummaryWriter(log_dir=log_dir)


# Load a pretrained model
model = YOLO("/workspace/ultralytics/cfg/models/11/ours_yaml/yolo11_smalldetect.yaml")

# Train the model on your custom dataset
model.train(data="/workspace/ultralytics/cfg/datasets/coco128.yaml", epochs=100 )