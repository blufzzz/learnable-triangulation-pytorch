import os
from collections import defaultdict
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox, check_black_border
from mvn.utils import volumetric
from mvn.datasets.utils import load_objects, get_skeleton, get_obj_transform, visualize_joints_2d
from IPython.core.debugger import set_trace
import argparse
import trimesh
from matplotlib import pyplot as plt
from PIL import Image

CAM_EXTR = np.array(
    [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
     [0, 0, 0, 1]])

CAM_INTR = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030], 
                     [0, 0, 1]])


REORDER_IDX = np.array([
    0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
    20
])

class FPHAB(Dataset):
    def __init__(self,
                 root='media/hpc-4_Raid/ibulygin/F-PHAB/',
                 table_name ='table.npy',
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 ignore_cameras=[],
                 crop=True,
                 use_equidistant_dataset=True,
                 **kwargs
                 ):

        self.root = root
        self.table_path = os.path.join(self.root, table_name)
        self.table = np.load(self.table_path).item()
                
        # how much there are consecutive frames in the sequence
        self.dt = kwargs['dt']
        # time dilation between frames
        self.dilation = kwargs['dilation']
        self.dilation_type = kwargs['dilation_type']
        self.keypoints_per_frame=kwargs['keypoints_per_frame']
        self.pivot_type = kwargs['pivot_type']
        self.custom_iterator = kwargs['custom_iterator']
        if self.custom_iterator is not None:
            assert self.dt == len(self.custom_iterator)

        if self.pivot_type == 'intermediate':
            assert self.dt==0 or self.dt//2 != 0, 'Only odd `dt` is supported for intermediate pivot!'

        # the whole time period covered, with dilation
        if self.dilation_type == 'exponential':
            self._time_period = np.exp(self.dilation+self.dt-2) + 1
        elif self.dilation_type == 'constant':
            self._time_period = self.dt + (self.dt-1)*(self.dilation)
        elif self.dilation_type == 'square':
            self._time_period = (self.dilation+self.dt-2)**2 + 1            
        else:
            raise RuntimeError('Wrong dilation_type') 

    def _initialize_table(self):
        pass

    def _unpack(self, shot, offset):
        
        subject = shot['subject_names']
        action = shot['action_names']
        frame_idx = shot['frame_idx'] 
        seq_idx = shot['seq_idx']
        keypoints = shot['keypoints']

        image_path = os.path.join(self.root, subject, action, seq_idx, 'color',
                                 'color_{:04d}.jpeg'.format(frame_idx + offset))

        image = cv2.imread(image_path)
        image = 2*((image / 255.0) - 0.5) # cast to pixes values to [-1,1] range from [0,255]    

        retval_camera = Camera(R, t, K) # TODO: create Camera object

        return image, keypoints, retval_camera

    def __getitem__(self, idx):
    
        sample = defaultdict(list) 
        shot = self.table[idx]

        # take shots that are consecutive in time, with specified pivot
        if self.pivot_type == 'intermediate':
            assert self.dilation_type == 'constant'
            iterator=range(-((self._time_period)//2), ((self._time_period)//2)+1, self.dilation+1)
        elif self.pivot_type == 'first':
            iterator={'constant':range(-(self._time_period-1), 1, self.dilation+1),
                      'custom': self.custom_iterator,  
                      'square':np.concatenate([[0], \
                               -(np.arange(self.dilation, self.dilation+self.dt-1)**2).astype(int)])[::-1]
                      }[self.dilation_type]
        else:
            raise RuntimeError('Unknown `pivot_type` in config.dataset.<train/val>')       
        assert 0 in iterator, 'iterator should contain 0 index for the pivot frame'    

        for i in iterator:    
            image, keypoints, retval_camera = self._unpack(shot, offset)

            if unpacked is not None:    
                #collect data from different cameras
                sample['images'].append(image)
                sample['cameras'].append(retval_camera)
                sample['proj_matrices'].append(retval_camera.projection)

                if self.keypoints_per_frame:
                    keypoints = np.pad(keypoints[:self.num_keypoints],
                                        ((0,0), (0,1)),
                                        'constant', constant_values=1.0)
                    
                    sample['keypoints_3d'].append(keypoints)
                    
                elif i == 0:         
                    sample['keypoints_3d'] = np.pad(keypoints[:self.num_keypoints],
                                                     ((0,0), (0,1)),
                                                     'constant', 
                                                     constant_values=1.0)
            
        sample['indexes'] = idx
        sample.default_factory = None
        return sample


    def __len__(self):
        return len(self.table)

    def evaluate(self, 
                keypoints_3d_predicted):
        raise RuntimeError
        keypoints_gt = self.table['keypoints']
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :] # TODO: define root index
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)

        return per_pose_error_relative.mean()