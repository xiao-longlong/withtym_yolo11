import os
import sys
os.getcwd()
sys.path.insert(0, os.getcwd())


from ultralytics import YOLO

# Load a model
model = YOLO("/workspace/runs/detect/train/weights/best.pt")  # load an official model

# Export the model
model.export(format="onnx")