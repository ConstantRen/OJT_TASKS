import torch
import cv2 #OCR Library
import numpy as np
import argparse
import time

print(torch.cuda.get_device_name(0))
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()
def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap

#LOAD OCR (yolov3,coco)
def load_yolo():
    net = cv2.dnn.readNetFromDarknet("yolov3.weights","yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_name = net.getLayerNames()
    output_layers= [layers_name[i-1] for i in net.getUnconnectedOutLayers()]
    colors = [(0, 255, 0) for _ in range(len(classes))]
    return net,classes,colors,output_layers


def detect_objects(img,net,outputLayers):
    blob = cv2.dnn.blobFromImage(img,scalefactor=0.00392, size=(480,480), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob,outputs

def get_box_dimensions(outputs,height,width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores= detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > 0.03:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes,confs,class_ids




def draw_labels(boxes,confs,colors,class_ids,classes,img):
    indexes = cv2.dnn.NMSBoxes(boxes,confs,0.5,0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
            cv2.putText(img,label,(x,y-5),font,1, color, 1)
    cv2.imshow("Image",img)


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, channels = frame.shape
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        
        # Put FPS text on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        cv2.imshow("Image", frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
	webcam = args.webcam
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()

	

	cv2.destroyAllWindows()











