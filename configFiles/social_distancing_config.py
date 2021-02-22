#path to YOLO directory
MODEL_PATH = "yolo-coco"

#initialize minimum porbability to filter 
#weak detections along with the threshold when apply non-maxim suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#bool to determine whether to use NVIDIA CUDA
USE_GPU = True

#define min safe distance that two people can be
# from each other
MIN_DISTANCE = 50

