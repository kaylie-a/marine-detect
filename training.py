from ultralytics import YOLO
import os

current_dir = os.getcwd()

# Load a model for training
model = YOLO()

# Train the model from dataset
# epoch: number of passes for training
# Output for training goes to runs\detect
model.train(data=current_dir+"\\datasets\\Images\\Squid\\Squid Image Dataset.v1i.yolov5pytorch\\data.yaml", epochs=100)
