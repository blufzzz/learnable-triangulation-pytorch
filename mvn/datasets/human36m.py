import os
from collections import defaultdict
import pickle

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils import volumetric


class Human36MMultiViewDataset(Dataset):
    """
        Human3.6M for multiview tasks.
    """
    def __init__(self,
                 h36m_root='/media/hpc2_storage/ibulygin/h36m-fetch/processed/',
                 labels_path='/media/hpc2_storage/ibulygin/human36m-preprocessing/human36m-multiview-labels-GTbboxes.npy',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 with_damaged_actions=False,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 undistort_images=False,
                 ignore_cameras=[],
                 crop=True,
                 **kwargs
                 ):
        """
            h36m_root:
                Path to 'processed/' directory in Human3.6M
            labels_path:
                Path to 'human36m-multiview-labels.npy' generated by 'generate-labels-npy-multiview.py'
                from https://github.sec.samsung.net/RRU8-VIOLET/human36m-preprocessing
            retain_every_n_frames_test:
                By default, there are 159 181 frames in training set and 26 634 in test (val) set.
                With this parameter, test set frames will be evenly skipped frames so that the
                test set size is `26634 // retain_every_n_frames_test`.
                Use a value of 13 to get 2049 frames in test set.
            with_damaged_actions:
                If `True`, will include 'S9/[Greeting-2,SittingDown-2,Waiting-1]' in test set.
            kind:
                Keypoint format, 'mpii' or 'human36m'
            ignore_cameras:
                A list with indices of cameras to exclude (0 to 3 inclusive)
        """
        assert train or test, '`Human36MMultiViewDataset` must be constructed with at least ' \
                              'one of `test=True` / `train=True`'
        assert kind in ("mpii", "human36m")

        self.h36m_root = h36m_root
        self.labels_path = labels_path
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.undistort_images = undistort_images
        self.ignore_cameras = ignore_cameras
        self.crop = crop

        self.labels = np.load(labels_path, allow_pickle=True).item()

        n_cameras = len(self.labels['camera_names'])
        assert all(camera_idx in range(n_cameras) for camera_idx in self.ignore_cameras)

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        indices = []
        if train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0])
        if test:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

            if not with_damaged_actions:
                mask_S9 = self.labels['table']['subject_idx'] == self.labels['subject_names'].index('S9')

                damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
                damaged_actions = [self.labels['action_names'].index(x) for x in damaged_actions]
                mask_damaged_actions = np.isin(self.labels['table']['action_idx'], damaged_actions)

                mask &= ~(mask_S9 & mask_damaged_actions)

            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

        self.num_keypoints = 16 if kind == "mpii" else 17
        assert self.labels['table']['keypoints'].shape[1] == 17, "Use a newer 'labels' file"

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]
            assert len(self.keypoints_3d_pred) == len(self)

    def __len__(self):
        return len(self.labels['table'])

    def _unpack(self, shot, camera_idx, camera_name, return_image_path = False):
        
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        # load bounding box
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]
        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            return None

        # scale the bounding box
        bbox = scale_bbox(bbox, self.scale_bbox)

        # load image
        image_path = os.path.join(
            self.h36m_root, subject, action, 'imageSequence' + '-undistorted' * self.undistort_images,
            camera_name, 'img_%06d.jpg' % (frame_idx+1))
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load camera
        shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

        if self.crop:
            # crop image
            image = crop_image(image, bbox)
            retval_camera.update_after_crop(bbox)

        image_shape_before_resize = None    
        if self.image_shape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, self.image_shape)
            retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

        if self.norm_image:
            image = normalize_image(image)

        bbox = bbox + (1.0,) # TODO add real confidences

        return image, bbox, retval_camera, image_shape_before_resize, image_path if return_image_path else None    

    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        shot = self.labels['table'][idx]

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue

            unpacked = self._unpack(shot, camera_idx, camera_name)
            if unpacked is None:
                continue
            else:        
                image, bbox, retval_camera, image_shape_before_resize, image_path = unpacked    

            sample['images'].append(image)
            sample['images_paths'].append(image_path)
            sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)
            if image_shape_before_resize is not None:
                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            
        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:self.num_keypoints],
            ((0,0), (0,1)), 'constant', constant_values=1.0)

        # build cuboid
        # base_point = sample['keypoints_3d'][6, :3]
        # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        # position = base_point - sides / 2
        # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # save sample's index
        sample['indexes'] = idx

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        sample.default_factory = None
        return sample

    def evaluate_using_per_pose_error(self, per_pose_error, split_by_subject):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(len(self.labels['action_names'])):
                action_mask = (self.labels['table']['action_idx'] == action_idx) & mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores[self.labels['action_names'][action_idx]] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            action_names_without_trials = \
                [name[:-2] for name in self.labels['action_names'] if name.endswith('-1')]

            for action_name_without_trial in action_names_without_trials:
                combined_score = {'total_loss': 0.0, 'frame_count': 0}

                for trial in 1, 2:
                    action_name = '%s-%d' % (action_name_without_trial, trial)
                    combined_score['total_loss' ] += action_scores[action_name]['total_loss']
                    combined_score['frame_count'] += action_scores[action_name]['frame_count']
                    del action_scores[action_name]

                action_scores[action_name_without_trial] = combined_score

            for k, v in action_scores.items():
                action_scores[k] = v['total_loss'] / v['frame_count']

            return action_scores

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        for subject_idx in range(len(self.labels['subject_names'])):
            subject_mask = self.labels['table']['subject_idx'] == subject_idx
            subject_scores[self.labels['subject_names'][subject_idx]] = \
                evaluate_by_actions(self, per_pose_error, subject_mask)

        return subject_scores

    def evaluate(self, keypoints_3d_predicted, split_by_subject=False, transfer_cmu_to_human36m=False, transfer_human36m_to_human36m=False):
        keypoints_gt = self.labels['table']['keypoints'][:, :self.num_keypoints]
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        if transfer_cmu_to_human36m or transfer_human36m_to_human36m:
            human36m_joints = [10, 11, 15, 14, 1, 4]
            if transfer_human36m_to_human36m:
                cmu_joints = [10, 11, 15, 14, 1, 4]
            else:
                cmu_joints = [10, 8, 9, 7, 14, 13]

            keypoints_gt = keypoints_gt[:, human36m_joints]
            keypoints_3d_predicted = keypoints_3d_predicted[:, cmu_joints]

        # mean error per 16/17 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)

        # relative mean error per 16/17 joints in mm, for each pose
        if not (transfer_cmu_to_human36m or transfer_human36m_to_human36m):
            root_index = 6 if self.kind == "mpii" else 6
        else:
            root_index = 0

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error, split_by_subject),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative, split_by_subject)
        }

        return result['per_pose_error_relative']['Average']['Average'], result



class Human36MSingleViewDataset(Human36MMultiViewDataset):
    """
        Human3.6M setup for singleview tasks (kawrgs['dt'] = 1) and other ones that exploits temporal information (kawrgs['dt'] > 1)
    """
    def __init__(self,
                 h36m_root='/media/hpc2_storage/ibulygin/h36m-fetch/processed/',
                 labels_path='/media/hpc2_storage/ibulygin/human36m-preprocessing/human36m-multiview-labels-GTbboxes.npy',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 with_damaged_actions=False,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 undistort_images=False,
                 ignore_cameras=[],
                 crop=True,
                 use_equidistant_dataset=True,
                 **kwargs
                 ):

        # Human36MSingleViewDataset, self
        super().__init__(h36m_root=h36m_root,
                         labels_path=labels_path,
                         pred_results_path=None,
                         image_shape=image_shape,
                         train=train,
                         test=test,
                         retain_every_n_frames_in_test=1,
                         with_damaged_actions=with_damaged_actions,
                         cuboid_side=cuboid_side,
                         scale_bbox=scale_bbox,
                         norm_image=norm_image,
                         kind=kind,
                         undistort_images=undistort_images,
                         ignore_cameras=ignore_cameras,
                         crop=crop,
                         use_equidistant_dataset = use_equidistant_dataset)


        # how much consecutive frames in the sequence
        self.dt = kwargs['dt']
        self.keypoints_per_frame=kwargs['keypoints_per_frame']
        self.pivot_type = kwargs['pivot_type']

        assert self.dt==0 or self.dt//2 != 0, 'Only ODD `dt` is supported!'
        # time dilation betweem frames
        self.dilation = kwargs['dilation']

        # optional, used train 278 line
        self.singleview = True
        self.evaluate_cameras = kwargs['evaluate_cameras']

        if test:
            self.iterate_cameras_names = [self.labels['camera_names'][camera_idx] for camera_idx in self.evaluate_cameras] 
        else:    
            self.iterate_cameras_names = [self.labels['camera_names'][camera_idx] for camera_idx in range(self.n_cameras) if camera_idx not in self.ignore_cameras]

        assert all(self.labels['camera_names'][camera_idx] in self.iterate_cameras_names for camera_idx in self.evaluate_cameras), 'Before evaluation on the cameras, iteration over the cameras should be done'

        n_frames = len(self.labels['table'])

        # Let's call an i-th element in the middle of the CONSECUTIVE sequence [i-dt//2, i, i+dt//2], with length `dt`, as pivot.
        # Initially, consider all frames as pivots
        pivot_mask = np.ones((n_frames,),dtype=np.bool)
        # the whole time period covered with dilation
        self._time_period = self.dt + (self.dt-1)*(self.dilation)

        if self.dt != 0:

            frame_idx =  self.labels['table']['frame_idx']
            
            # Mark positions where the new scene or action is starting
            change_mask = np.concatenate((frame_idx[:-1] > frame_idx[1:], [False]))

            if self.pivot_type == 'intermediate':
                # Shift that positions shuch that all non-pivot positions marked `True`
                for _ in range((self._time_period//2)-1):
                    change_mask[:-1] = change_mask[:-1] | change_mask[1:]
                for _ in range(self._time_period//2):
                    change_mask[1:] = change_mask[1:] | change_mask[:-1]
                change_mask[:self._time_period//2] = True
                change_mask[-(self._time_period//2):] = True

            elif self.pivot_type == 'first':
                # Shift that positions shuch that all non-pivot positions marked `True`
                for _ in range(_time_period-1):
                    change_mask[1:] = change_mask[1:] | change_mask[:-1]
                change_mask[:_time_period-1] = True

            else:
                raise RuntimeError('Unknown `pivot_type` in config.dataset.<train/val>')   


            pivot_mask = ~change_mask

        self.pivot_mask = pivot_mask
        self.pivot_indxs = np.arange(n_frames)[pivot_mask][::retain_every_n_frames_in_test]
        self.n_sequences = len(self.pivot_indxs)

    def __getitem__(self, idx):

        camera_idx = idx // self.n_sequences
        shot_idx = idx % self.n_sequences
        pivot_idx = self.pivot_indxs[shot_idx]
        camera_name = self.iterate_cameras_names[camera_idx]

        sample = defaultdict(list) # return value

        # take shots that are consecutive in time, with specified pivot
        if self.pivot_type == 'intermediate':
            iterator=range(-((self._time_period)//2), ((self._time_period)//2)+1, self.dilation+1)
        elif self.pivot_type == 'first':
            iterator=range(-(self._time_period-1),1)
        else:
            raise RuntimeError('Unknown `pivot_type` in config.dataset.<train/val>')       

        for i in iterator:
            
            shot = self.labels['table'][pivot_idx+i]
            unpacked = self._unpack(shot, camera_idx, camera_name)
            if unpacked is None: # we need full sequence
                return None
            else:        
                image, bbox, retval_camera, image_shape_before_resize, image_path = unpacked

            if unpacked is not None:    
                #collect data from different cameras
                sample['images'].append(image)
                sample['images_paths'].append(image_path)
                sample['detections'].append(bbox) 
                sample['cameras'].append(retval_camera)
                sample['proj_matrices'].append(retval_camera.projection)

                if self.keypoints_per_frame:
                    keypoints = np.pad(shot['keypoints'][:self.num_keypoints],((0,0), (0,1)),
                                             'constant', constant_values=1.0)
                    sample['keypoints_3d'].append(keypoints)
                    
                if image_shape_before_resize is not None:
                    sample['image_shapes_before_resize'].append(image_shape_before_resize)

        if not self.keypoints_per_frame:         
            pivot_shot = self.labels['table'][pivot_idx]
            sample['keypoints_3d'] = np.pad(pivot_shot['keypoints'][:self.num_keypoints],((0,0), (0,1)),
                                             'constant', constant_values=1.0)
            # build cuboid
            # base_point = sample['keypoints_3d'][6, :3]
            # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            # position = base_point - sides / 2
            # sample['cuboids'] = volumetric.Cuboid3D(position, sides)
            # save sample's index
            
        sample['indexes'] = idx
        sample.default_factory = None
        return sample


    def __len__(self):
        return self.n_sequences*len(self.iterate_cameras_names)

    def evaluate(self, 
                keypoints_3d_predicted,
                result_indexes,
                mask = None,
                split_by_subject=False,
                transfer_cmu_to_human36m=False, 
                transfer_human36m_to_human36m=False):

        original_labels = self.labels['table'].copy()

        cameras_results  = {}
        # get indexes corrsesponding to cameras we've iterated over 
        # e.g. we've iterated over cameras [1,2], but in the `__getitem__` they've [0,1] `camera_idx` 
        evaluate_cameras_indexes = [self.iterate_cameras_names.index(self.labels['camera_names'][camera_idx]) for camera_idx in self.evaluate_cameras]
        for camera_index in evaluate_cameras_indexes:

            # choose indexes corresponding to the camera
            camera_mask = result_indexes // self.n_sequences == camera_index
            indexes_for_camera = result_indexes[camera_mask]
            shot_indexes_for_camera = indexes_for_camera % self.n_sequences

            # to ensure proper evaluation in super().evaluate() below
            self.labels['table'] = original_labels[self.pivot_indxs][shot_indexes_for_camera]

            result = super(Human36MSingleViewDataset, self).evaluate(keypoints_3d_predicted[camera_mask],
                                                                    mask = mask,
                                                                    split_by_subject=split_by_subject,
                                                                    transfer_cmu_to_human36m=transfer_cmu_to_human36m,
                                                                    transfer_human36m_to_human36m=transfer_human36m_to_human36m)

            cameras_results[camera_index] = result

        # to ensure furtfer __getitem__ iterations
        self.labels['table'] = original_labels

        return cameras_results