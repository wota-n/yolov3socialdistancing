#import necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0): #personIdx to filter out all objects as humans since we want to detect humans
    #grab dimension of frame and init list of
    #results
    (H, W) = frame.shape[:2]
    results = []

    #construct a blob from input from followed by performing a forward
    #pass of the YOLO object detector, creating the bounidng boxes
    #and assosciated probabilities. blobs: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    #initialize lists of detected bounding boxes, centroids and 
    #confidence in that order
    boxes = []
    centroids = []
    confidences = []

    #loop over each layer output
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            #extract class ID and confidence (referring to probability)
            #of the current detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter detections by (1) ensuring detected objects
            #was a person and (2) that the minimum confidence is
            #met
            if classID == personIdx and confidence > MIN_CONF:
                #scales bounding box coordinates relative to
                #size of image. However, YOLo actually returns center (x,y)
                #coordinates the bounding box followed by the boxes' width
                #and height
                
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int") 
                #astype: https://www.askpython.com/python/built-in-methods/python-astype

                #use center (x, y)- coordinates to derive the top
                #and left corner of bounding box
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                #update list of bounding box coordinates,
                #centroids, and confidence
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #applying non-maxima suppression to suppress weak and overlapping
    #bounding boxes cv2.dnn.NMSBoxes is a built in method from OpenCV
    nms_detect = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    #ensure at least one detection exists
    if len(nms_detect) > 0:
        #loop over the indexes we are keeping
        for i in nms_detect.flatten():
            #extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #update results list to consist of person
            #prediction probability, bounding box coordinates,
            #and the centroid
            r = (confidences[i], (x, y, x + w, y+h), centroids[i])
            results.append(r)

    # return the list of results
    return results
            
                