import pickle
import os
import numpy as np
import cv2
import time
import sys
import re
from copy import deepcopy
from collections import defaultdict
from itertools import islice
from tqdm import tqdm_notebook, tqdm
from time import time
from easydict import EasyDict
from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt
from warnings import filterwarnings

import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader


from mvn.utils.img import image_batch_to_numpy, denormalize_image,to_numpy
from mvn.models.triangulation import VolumetricTriangulationNet
from mvn.models.pose_hrnet import get_pose_net as get_pose_hrnet
from mvn.models.pose_resnet import get_pose_net as get_pose_resnet

from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion
from mvn.utils.vis import draw_2d_pose
from mvn.utils import img
from mvn.utils import multiview
from mvn.utils import volumetric
from mvn.utils import op
from mvn.utils import vis
from mvn.utils import cfg
from mvn.utils.misc import get_start_stop_frame_indxs, index_to_name, get_error_diffs, normalize_temporal_images_batch, retval
from mvn.datasets import utils as dataset_utils
from mvn.datasets.human36m import Human36MMultiViewDataset, Human36MTemporalDataset

from train import setup_human36m_dataloaders

from mvn.models.volumetric_adain import VolumetricTemporalAdaINNet
from mvn.models.volumetric_grid import VolumetricTemporalGridDeformation

from IPython.core.debugger import set_trace




JOINT_H36_DICT = {0:'RFoot',
                 1:'RKnee',
                 2:'RHip',
                 3:'LHip',
                 4:'LKnee',
                 5:'LFoot',
                 6:'Hip',
                 7:'Spine',
                 8:'Thorax',
                 9:'Head',
                 10:'RWrist',
                 11:'RElbow',
                 12:'RShoulder',
                 13:'LShoulder',
                 14:'LElbow',
                 15:'LWrist',
                 16:'Neck/Nose'}

JOINT_NAMES_DICT = {
                    0: "nose",
                    1: "left_eye",
                    2: "right_eye",
                    3: "left_ear",
                    4: "right_ear",
                    5: "left_shoulder",
                    6: "right_shoulder",
                    7: "left_elbow",
                    8: "right_elbow",
                    9: "left_wrist",
                    10: "right_wrist",
                    11: "left_hip",
                    12: "right_hip",
                    13: "left_knee",
                    14: "right_knee",
                    15: "left_ankle",
                    16: "right_ankle"
                }


CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}


device = 'cuda:0' #torch.cuda.current_device()

##########
# DEFINE #
##########

experiment_type='resnet_50/spade_3d'
experiment_name='h36_sv32_dist_spade3d-128-interpolate-gn_s2v-v2v-256-32-gn_vf32_f2v-features-noupscale-C0-256-group_resnet50-gn-nostylegrad_dt-12_dil-3-1-1_lr-1e-4@20.03.2020-14:31:23'

# experiment_type = 'resnet_50/stack_3d'
# experiment_name='h36_sv32_dist_stack3d-interpolate-gn_s2v-v2v-256-256-64-gn_vf64_f2v-backbone-noupscale-C0-256-group_resnet50-gn-nostylegrad_dt-12_dil-3-1-1_lr-1e-4@19.03.2020-23:02:34'

# experiment_type = 'resnet_50/stack_2d'
# experiment_name = 'h36_sv32_dist_stack-unproject-gn_s2v-lstm2d-256-128-64-gn_vf32-v2v-conf_f2v-backbone-noupscale-C4-256-group_resnet50-gn-nostylegrad_dt-12_dil-3-1-1_lr-1e-4@12.03.2020-22:37:17'

# experiment_type = 'resnet_50/adain_1d'
# experiment_name = 'h36_sv32_dist_adain-all-gn_s2v-lstm-1024-1024-1024-gn_vf32_f2v-backbone-C4-1024-group_resnet50-gn-nostylegrad_dt-12_dil-3-1-1_lr-1e-4@12.03.2020-19:00:21'     

# experiment_type = 'resnet_50/baseline'
# experiment_name = 'h36_sv32_dist_resnet50-gn_v2v-v1-gn-no-aggr_1-1_lr-1e-4_boneloss-1e-4@19.03.2020-19:21:45'


length = 250
subjects = ['S9']
actions = ['Directions-1', 'Greeting-1', 'Smoking-1', 'Walking-1']

USE_RANDOM_STYLE_VECTOR = True
USE_CONSTANT_STYLE_VECTOR = True
RETURN_STYLE_VECTOR = True
ADD_IMAGES = True

add_per_joint_description = True


#########
# START #
#########

experiment_root = os.path.join('../logs/', experiment_type, experiment_name)
config_path = experiment_root + '/config.yaml'

config = cfg.load_config(config_path)
config.dataset.val.retain_every_n_frames_in_test = 1

_, val_loader, _ = setup_human36m_dataloaders(config,
                                             is_train=False,
                                             distributed_train=False)

batch_size, dt, dilation = val_loader.batch_size, val_loader.dataset.dt, val_loader.dataset.dilation

pivot_type = config.dataset.pivot_type 
keypoints_per_frame = config.dataset.val.keypoints_per_frame if hasattr(config.dataset.val, 'keypoints_per_frame') else False
pivot_position = {'first':-1, 'intermediate':dt//2}[pivot_type]

if not hasattr(config.model.backbone, 'group_norm'):
    config.model.backbone.group_norm = False

print('Batch size:', batch_size, '\n',\
      'dt:', dt, '\n',\
      'dilation:', dilation, '\n',\
      'pivot_type:', pivot_type, '\n',\
      'pivot_position:', pivot_position,'\n',\
      'keypoints_per_frame', keypoints_per_frame)


model = {
    "vol": VolumetricTriangulationNet,
    "vol_temporal_adain":VolumetricTemporalAdaINNet
}[config.model.name](config, device=device).to(device)

print ('LOADED {} MODEL'.format(config.model.name))


checkpoints_path = experiment_root + '/checkpoints/'
weights_path = checkpoints_path + '/weights.pth'
model.load_state_dict(torch.load(weights_path)['model_state'], strict=True)
print ('STATE DICT LOADED')


labels=val_loader.dataset.labels
start_frame_indexes, stop_frame_indxs=get_start_stop_frame_indxs(labels)

for subject in subjects:
    for action in actions:

        action_index=retval['action_names'].index(action)
        offset = len(retval['action_names']) if subject == 'S11' else 0
        start=start_frame_indexes[offset:][action_index].item()

        assert length%batch_size==0
        assert subject in ['S9', 'S11']
        assert subject, action == index_to_name(start+length, stop_frame_indxs)


        KIND = model.kind
        series = defaultdict(list)
        series['images'] = defaultdict(list)
        series['proj_matrices'] = defaultdict(list)
        eval_view = 0 # supported [0,1,2,3] cameras
        collate_fn = dataset_utils.make_collate_fn(randomize_n_views=False,
                                                   min_n_views=None,
                                                   max_n_views=None)


        model.eval()
        with torch.no_grad():
            for i in tqdm(range(start,start+length)):
                ##############
                # EVALUATION #
                ##############
                batch = val_loader.dataset.__getitem__(i + eval_view*val_loader.dataset.n_sequences)
                if batch is None:
                    'Batch is none...'
                    break
                
                batch = collate_fn([batch])
                (images_batch, 
                keypoints_3d_gt, 
                keypoints_3d_validity_batch_gt, 
                proj_matricies_batch) = dataset_utils.prepare_batch(batch, device)
                
                if RETURN_STYLE_VECTOR:
                    output = model(images_batch, batch, return_style_vector=True)
                    series['style_vectors'].append(output[-1].detach().cpu().numpy())
                else:    
                    output = model(images_batch, batch)
                torch.cuda.empty_cache()
                
                if USE_RANDOM_STYLE_VECTOR:
                    randomized_output = model(images_batch, batch, randomize_style=True)
                    keypoints_3d_pred_random_style = randomized_output[0]
                    keypoints_random_style = keypoints_3d_pred_random_style.detach().cpu().numpy()
                    series['keypoints_random_style'].append(keypoints_random_style)
                    torch.cuda.empty_cache()
                    
                if USE_CONSTANT_STYLE_VECTOR:
                    const_output = model(images_batch, batch, const_style_vector=True, return_style_vector=True)
                    keypoints_3d_pred_const_style = const_output[0]
                    keypoints_const_style = keypoints_3d_pred_const_style.detach().cpu().numpy()
                    series['keypoints_const_style'].append(keypoints_const_style)
                    series['style_vectors_const'].append(const_output[-1].detach().cpu().numpy())
                    torch.cuda.empty_cache()    

                keypoints_3d_pred = output[0]
                batch_size, n_views, n_joints = keypoints_3d_gt.shape[:3]

                # normalize all stuff
                proj_matricies_batch = proj_matricies_batch[:,pivot_position].detach().cpu().numpy()
                images_batch = normalize_temporal_images_batch(images_batch, pivot_position)
                keypoints = keypoints_3d_pred.detach().cpu().numpy()
                keypoints_gt =  keypoints_3d_gt.detach().cpu().numpy()
                
                series['keypoints'].append(keypoints)
                series['keypoints_gt'].append(keypoints_gt)
                
                if ADD_IMAGES:
                    series['images'][eval_view].append(images_batch)
                    series['proj_matrices'][eval_view].append(proj_matricies_batch)
                
                ##############
                # PROJECTING #
                ##############
                for view_i in range(4):
                    if view_i == eval_view:
                        continue
                    else:
                        batch = val_loader.dataset.__getitem__(i + view_i*val_loader.dataset.n_sequences)
                        batch = collate_fn([batch])
                        
                        (images_batch, 
                        keypoints_3d_gt, 
                        keypoints_3d_validity_batch_gt, 
                        proj_matricies_batch) = dataset_utils.prepare_batch(batch, device)
                        
                        # normalize all stuff
                        proj_matricies_batch = proj_matricies_batch[:,pivot_position].detach().cpu().numpy()
                        images_batch = normalize_temporal_images_batch(images_batch, pivot_position)
                        
                        if ADD_IMAGES:
                            series['images'][view_i].append(images_batch)
                            series['proj_matrices'][view_i].append(proj_matricies_batch)
                
        series['keypoints'] = np.concatenate(series['keypoints'],0)
        series['keypoints_gt'] = np.concatenate(series['keypoints_gt'],0)
        if USE_RANDOM_STYLE_VECTOR:
            series['keypoints_random_style'] = np.concatenate(series['keypoints_random_style'],0)
        if RETURN_STYLE_VECTOR:
            series['style_vectors'] = np.concatenate(series['style_vectors'],0) 
        if USE_CONSTANT_STYLE_VECTOR:
            series['keypoints_const_style'] = np.concatenate(series['keypoints_const_style'],0) 
            series['style_vectors_const'] = np.concatenate(series['style_vectors_const'],0)   
        if ADD_IMAGES:    
            for i in range(4):
                series['images'][i] = np.concatenate(series['images'][i],0)
                series['proj_matrices'][i] = np.concatenate(series['proj_matrices'][i],0)


        # check const-style-vectors
        if USE_CONSTANT_STYLE_VECTOR:
            flag = []
            for i in range(length):
                for j in range(length): 
                    flag += [(series['style_vectors_const'][j] == series['style_vectors_const'][i]).all()]
            print ('All style_vectors_const are equal', all(flag))         


        #########
        # ERROR #   
        #########
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
        error, gt_diffs, model_diffs = get_error_diffs(series['keypoints_gt'], series['keypoints'])
        ax1.plot(error, 'g', label='Error')
        title = f"Error: {error.mean()} \n"

        if USE_CONSTANT_STYLE_VECTOR:
            error_const_style, _, model_diffs_const_style = get_error_diffs(series['keypoints_gt'], series['keypoints_const_style'])
            ax1.plot(error_const_style, 'red', label='Error')
            title += f"Error_const_style: {error_const_style.mean()} \n"

        if USE_RANDOM_STYLE_VECTOR:
            error_rand_style, _, model_diffs_rand_style = get_error_diffs(series['keypoints_gt'], series['keypoints_random_style'])
            ax1.plot(error_rand_style, 'orange', label='Error')
            title += f"Error_rand_style: {error_rand_style.mean()} \n"

        ax1.set_xlabel('№ frame')
        ax1.set_ylabel('MPJPE')
        ax1.set_title(title)

        #########
        # DIFFS #   
        #########
        ax2.plot(model_diffs, 'g',label='Model diffs')

        if USE_RANDOM_STYLE_VECTOR:
            ax2.plot(model_diffs_rand_style, 'orange',label='Model RANDOM_STYLE diffs')
        if USE_CONSTANT_STYLE_VECTOR:
            ax2.plot(model_diffs_const_style, 'red',label='Model CONST_STYLE diffs')    

        ax2.plot(gt_diffs, label='GT diffs')
        ax2.set_xlabel('№ frame')
        ax2.set_ylabel(r'$RMSE(keyp_{i}, keyp_{i-1})$')
        ax2.legend()
        plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_err_diffs'), dpi=280)


        #########
        # STYLE #   
        #########
        if RETURN_STYLE_VECTOR:
            style_vectors = series['style_vectors']
            plt.figure(figsize=(10,10))
            plt.hist(style_vectors.flatten(), bins=100)
            plt.yscale('log', nonposy='clip')
            plt.xlabel('Value')
            plt.ylabel('Number')

            plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_hist'), dpi=280)
            plt.close('all')


            n_frame = 0
            time_step = 0

            if experiment_type[-2:] == '2d':
                n_style_channels = style_vectors.shape[-3]
                n = int(np.sqrt(n_style_channels))
                fig, axes = plt.subplots(ncols=n, nrows=n, figsize=(n*5,n*5))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(style_vectors[n_frame, 0, i])
                

                plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_style_feature_maps_frame={n_frame}'), dpi=280)        
                plt.close('all')
                    
                    
            elif experiment_type[-2:] == '3d':
                n_style_channels = style_vectors.shape[-4]
                n = int(np.sqrt(n_style_channels))
                fig, axes = plt.subplots(ncols=n, nrows=n, figsize=(n*5,n*5))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(style_vectors[n_frame, i, time_step])  
                

                plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_style_volume_slice_t={time_step}_frame={n_frame}'), dpi=280)       
                plt.close('all')


            ptp = style_vectors.reshape(length, -1).ptp(0)
            var = style_vectors.reshape(length, -1).var(0)    

            #######
            # PTP #
            #######

            plt.figure(figsize=(10,10))
            plt.plot(ptp)
            plt.title(f'PTP mean: {ptp.mean()}')
            plt.ylabel('mean ptp over time')
            plt.xlabel('feature index')
            plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_ptp'), dpi=280)    
            plt.close('all')


            #######
            # VAR #
            #######

            plt.figure(figsize=(10,10))
            plt.plot(var)
            plt.title(f'VAR mean: {var.mean()}')
            plt.ylabel('mean variance over time')
            plt.xlabel('feature index')
            plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_var'), dpi=280)    
            plt.close('all')

            
            #########
            # DIFFS #
            #########
                    
            style_diffs = []
            for i in range(1,length):
                style_diffs.append(np.linalg.norm(style_vectors[i] - style_vectors[i-1]))
            plt.figure(figsize=(10,10))
            plt.plot(style_diffs) 
            plt.xlabel('№ frame')
            plt.ylabel(r'$||s(t_i) - s(t_{i-1}) ||$')

            plt.savefig(os.path.join(experiment_root, f'{subject}_{action}_style_diffs'), dpi=280)     
            plt.close('all')
               


        #########
        # VIDEO #
        #########

        video_path = os.path.join(experiment_root, 'videos')
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
            print ('video_path - Created')
        else:
            print ('video_path - Already exists')
            
        keypoints_dir = os.path.join(video_path,'{0}_{1}'.format(subject, action),'keypoints_videos_lowres')
        if not os.path.isdir(keypoints_dir):
            os.makedirs(keypoints_dir)
            print ('keypoints_dir - Created')
        else:
            print ('keypoints_dir - Already exists')


        for i in tqdm(range(length)):
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (45,45))
            for view in range(4):
                
                # unpack
                image = series['images'][view][i]
                proj_matrice = series['proj_matrices'][view][i]
                keypoints_3d_gt = series['keypoints_gt'][i]
                keypoints_3d_pred = series['keypoints'][i]
                
                ax_i = ax.flatten()[view]
                ax_i.imshow(image)
                
                pjpe = np.linalg.norm(keypoints_3d_gt - keypoints_3d_pred, axis=-1)
                
                # predicted keypoints
                keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(proj_matrice,
                                                                                            keypoints_3d_pred)
                draw_2d_pose(keypoints_2d_pred_proj,ax_i,kind='human36m', point_size=200, line_width=5)

                keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(proj_matrice,
                                                                                            keypoints_3d_gt)
                draw_2d_pose(keypoints_2d_gt_proj, ax_i,kind='human36m', point_size=200, line_width=5, color='g')
                    
                if view == eval_view and add_per_joint_description:
                    ax_i.set_title('EVAL_VIEW', fontsize=34)
                    text = ''.join([j_name + f': ~{int(pjpe[j_n])}' + '\n' for j_n,j_name in JOINT_H36_DICT.items()])
                    h,w = image.shape[:2]
                    offset_1, offset_2 = 5,10
                    ax_i.text(0+offset_1, h-offset_2, text, style='italic', fontsize=25,
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                
                plt.savefig('./{}/img_{:05}.jpg'.format(keypoints_dir, i), bbox_inches='tight', dpi=100)
        


        # os.system("ffmpeg -framerate 25 -i keypoints_videos_lowres/img_%05d.jpg -vcodec mpeg4 -y {6}/movie_{0}_{1}_{2}_f{3}_{4}_{5}.mp4".format(*index_to_name(start+length, stop_frame_indxs),
                                                                                                                                                               length,
                                                                                                                                                               video_path))        