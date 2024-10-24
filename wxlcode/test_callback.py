import os
import sys
os.getcwd()
sys.path.insert(0, os.getcwd())

from ultralytics import YOLO

def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch

    # Ensure that image is a list
    image = image if isinstance(image, list) else [image]

    # Combine the prediction results with the corresponding frames
    predictor.results = zip(predictor.results, image)


# Create a YOLO model instance
model = YOLO("yolo11n.pt")

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for result, frame in model.predict():  # or model.track()
    pass