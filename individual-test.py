# For individual models

import os
import glob
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_width = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_height = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

# Parse ground truth labels
def parse_ground_truth(lbl_path):
    with open(lbl_path, "r") as label_file:
        lines = label_file.readlines()
        ground_truth = []
        for line in lines:
            label_data = line.strip().split()
            class_label = int(label_data[0])  # Class label
            bbox = [float(x) for x in label_data[1:]]  # Bounding box (x_center, y_center, width, height)
            ground_truth.append({"class": class_label, "bbox": bbox})
    return ground_truth

# Get predictions from model
def get_predictions(image, model):
    results = model.predict(source=image, save=False, verbose=False)
    predictions = []
    for box in results[0].boxes:
        class_label = int(box.cls.item())  # Class index
        confidence = float(box.conf.item())  # Confidence score
        bbox = box.xywhn.tolist()[0]  # Bounding box in normalized format [x_center, y_center, width, height]
        predictions.append({"class": class_label, "confidence": confidence, "bbox": bbox})
    return predictions

# Match predictions to ground truth
def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    matched = []
    for gt in ground_truth:
        for pred in predictions:
            iou = calculate_iou(gt["bbox"], pred["bbox"])
            if iou > iou_threshold and gt["class"] == pred["class"]:
                matched.append((gt, pred))
                break  # Each ground truth box can only be matched once
    return matched

# Evaluate metrics
def evaluate_metrics(predictions, ground_truth, iou_threshold=0.5):
    matched = match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold)

    # True Positives: Matched predictions
    true_positives = len(matched)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0

    return precision, recall, f1, accuracy

# Evaluate model on dataset
def evaluate_model_on_dataset(model_path, image_files, label_files, iou_threshold=0.5):
    # Load model
    model = YOLO(model_path)

    # Initialize metrics
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_accuracies = []

    # Loop through images and labels
    for img_path, lbl_path in zip(image_files, label_files):
        ground_truth = parse_ground_truth(lbl_path)
        image = Image.open(img_path).convert("RGB")
        predictions = get_predictions(image, model)

        # Evaluate metrics for this image
        precision, recall, f1, accuracy = evaluate_metrics(predictions, ground_truth, iou_threshold)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(accuracy)

    # Calculate average metrics
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)

    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall:    {avg_recall:.2f}")
    print(f"Average F1 Score:  {avg_f1:.2f}")
    print(f"Average Accuracy:  {avg_accuracy:.2f}")

current_dir = os.getcwd()

image_files = sorted(glob.glob(current_dir+"\\datasets\\Images\\Clam\\(Mussell)mollusk.v2i.yolov5pytorch\\test\\images\\*.jpg"))
label_files = sorted(glob.glob(current_dir+"\\datasets\\Images\\Clam\\(Mussell)mollusk.v2i.yolov5pytorch\\test\\labels\\*.txt"))
model = current_dir+"\\models\\clam_model.pt",

evaluate_model_on_dataset(model, image_files, label_files, iou_threshold=0.5)
