from ultralytics import YOLO
import os

current_dir = os.getcwd()

# Load a pretrained model (optional)
model = YOLO() 

# Train the model on your custom dataset
model.train(data=current_dir+"\\datasets\\Images\\Dolphin\\(Common Dolphin)dophin.v1i.yolov5pytorch\\data.yaml", epochs=100)