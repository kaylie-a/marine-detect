from ultralytics import YOLO
import os

current_dir = os.getcwd()

# Load a pretrained model (optional)
model = YOLO(current_dir+"\\models\\clownfish_model.pt")

# Train the model on your custom dataset
model.train(data=current_dir+"\\datasets\\Images\\Fish\\(Clownfish)-Amphiprion-percula-.v3i.yolov5pytorch\\data.yaml", epochs=50)
