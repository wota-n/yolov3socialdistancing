#import packages
from configFiles import social_distancing_config as config
from configFiles.detection import detect_people
from scipy.spatial import distance as dist #used to determine the Euclidean distance
import numpy as np
import argparse
import imutils
import cv2
import os
import time

#construct arugment parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, default='', 
    help="path to (optional) input video file")
ap.add_argument('-o', '--output', type=str, default='', 
    help="path to (optional) output video file")
ap.add_argument('-d', '--display', type=int, default=1, 
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

#load COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#derive the paths to the YOLO weights and model config
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

#load YOLO object detector trained on COCO dataset (contains 80 classes)
print ("[INFO] loading YOLO from disk...")
tic = time.perf_counter()
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#check for GPU usage
if config.USE_GPU:
    #set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#determine only "output" layer names that are neeeded from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1 ] for i in net.getUnconnectedOutLayers()]

#initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer=None

#loop over frames from video stream
while True:
    #read next frame from the file
    (grabbed, frame) = vs.read()
    print("[INFO] processing video")

    #if no frame is grabbed, means video is finished
    #therefore we break
    if not grabbed:
        toc = time.perf_counter()

        print(f"Video processing took {toc - tic:0.4f} seconds")
        break

    #resize frame and detect ONLY people/humans
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,
        personIdx=LABELS.index("person"))

    #initialize the set of indexes that violate the minimum social
    #distance
    violate = set()

    #ensure the detection is of at least TWO persons
    #this is required for the system to compute the pairwise distance maps
    if len(results) >= 2:
        #extracts centroids from results and compute the
        #Euclidean distances between all pairs of centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        #loop over the upper triangular of distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                #check to see if distance between any two
                #centroid pairs is less than configured number
                #of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    #update violation set with index of
                    #centroid pairs
                    violate.add(i)
                    violate.add(j)

    #loop over results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        #extract the bounding box and centroid coordinates, then
        #initalize color of annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        #if the index pair exists within violation set,
        #update color
        if i in violate:
            color = (0, 0, 255)

        #draw (1) a bounding box around the person and (2)
        #the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    #draw total number of social distancing violations on the 
    #output frame
    text = "Detected Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 3)
        
    #check to see if output frame should be displayed to
    #screen
    if args["display"] > 0:
        #show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #if 'q' is pressed, break from loop
        if key == ord("q"):
            break

    #if an output video file path is supplied
    #and video writer is not init, do now    
    if args["output"] != "" and writer is None:
        #init video writer
        fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
            (frame.shape[1], frame.shape[0]), True) 

        #if the video writer is not None, write frame to output
        #video file
    if writer is not None:
        writer.write(frame)


