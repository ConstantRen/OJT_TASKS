import torch
import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=True)
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

# Load COCO names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov10x.pt')  # Load a pre-trained YOLOv10 model
model.to(device)
model.eval()

def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap

def detect_objects_pytorch(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model([img_rgb])  

    detections = results[0].boxes.data.cpu().numpy()

    return detections

def draw_labels_pytorch(detections, img):
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = classes[int(cls)]
        color = (0, 255, 0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f'{label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow('Image', img)

def webcam_detect():
    cap = start_webcam()

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        detections = detect_objects_pytorch(frame)
        draw_labels_pytorch(detections, frame)

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Image', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam = args.webcam
    verbose = args.verbose
    if webcam:
        if verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()
