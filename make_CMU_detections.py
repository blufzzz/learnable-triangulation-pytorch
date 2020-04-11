import sys
sys.path.append('/media/hpc2_storage/ibulygin/learnable-triangulation-pytorch/PyTorch_YOLOv3/')
from PyTorch_YOLOv3.models import Darknet
from PyTorch_YOLOv3.utils.utils import non_max_suppression, rescale_boxes, load_classes
from torchvision.transforms import Resize, ToTensor, Compose
import time
import datetime
from PIL import Image
import os 
import numpy as np
import torch
from tqdm import tqdm
import json
from collections import defaultdict
from IPython.core.debugger import set_trace


def check_detections(detections):

    return detections[0] is not None

device = 'cuda:0'
config = 'PyTorch_YOLOv3/config/yolov3.cfg'
weights_path = 'PyTorch_YOLOv3/weights/yolov3.weights'
model = Darknet(config).to(device)
model.load_darknet_weights(weights_path)

class_thresh = 0.9
img_size=416
conf_thres = 0.01
nms_thres = 0.5

cmu_root = '/media/hpc3_storage/ibulygin/panoptic-toolbox/'
poses = ['171026_pose1',
        '171026_pose2',
        '171026_pose3',
        '171204_pose1',
        '171204_pose2',
        '171204_pose3',
        '171204_pose4',
        '171204_pose5',
        '171204_pose6']



classes = load_classes("PyTorch_YOLOv3/data/coco.names")  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for pose in tqdm(poses):
    pose_dir = os.path.join(cmu_root, pose)
    images_dir = os.path.join(pose_dir, 'hdImgs')
    detections_dict = defaultdict(dict)
    detections_dict_path = os.path.join('/media/hpc2_storage/ibulygin/CMU_data/',pose ,'yolo-detections')
    
    for camera_name in os.listdir(images_dir):
        images_camera_dir = os.path.join(images_dir, camera_name)
        for image_name in sorted(os.listdir(images_camera_dir)):
            image_path =  os.path.join(images_camera_dir, image_name)

            # Configure input
            pil_image = Image.open(image_path)
            img = np.asarray(pil_image)
            transforms = Compose([Resize((img_size,img_size)), ToTensor()])
            x_tensor = transforms(pil_image).unsqueeze(0).to(device)

            # Get detections
            with torch.no_grad():
                detections = model(x_tensor)
                detections = non_max_suppression(detections, conf_thres, nms_thres)

            # Draw bounding boxes and labels of detections
            # Rescale boxes to original image
            if check_detections(detections):
                detections = rescale_boxes(detections[0], img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)

                # sort detections
                is_person = detections[:,-1] == 0
                person_class_confidence = detections[:,5] >= class_thresh
                if (is_person * person_class_confidence).any():
                    # FOUND
                    person_detections = detections[detections[:,-1] == 0]
                    person_detections_conf = person_detections[person_detections[:,5] >= class_thresh]
                    detections = person_detections_conf[person_detections_conf[:,4].argmax()]
                    detections = detections.tolist()
                else:
                    detections = None
            
            #  x1, y1, x2, y2, conf, cls_conf, cls_pred
            detections_dict[camera_name][image_name] = detections

        os.makedirs(detections_dict_path, exist_ok=True)    
        with open(f'{detections_dict_path}/{camera_name}.json', 'w') as fp:
            json.dump(detections_dict, fp)        
                
    

         