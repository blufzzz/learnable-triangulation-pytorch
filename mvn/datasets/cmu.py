import os
import json
from collections import defaultdict
from copy import deepcopy
import pickle

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, scale_bbox, resize_image, crop_image, image_batch_to_torch, normalize_image
from mvn.utils import volumetric

class CMUMultipleSceneDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumsum = [0] + list(np.cumsum([len(dataset) for dataset in self.datasets]))

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        dataset_i = np.digitize(idx, self.cumsum) - 1
        return self.datasets[dataset_i].__getitem__(idx - self.cumsum[dataset_i], global_idx=idx)

    def evaluate(self, results):
        scalar_metric = 0.0
        full_metric = {}
        return scalar_metric, full_metric


class CMUSceneDataset(Dataset):
    """Dataset for single CMU scene focused on multiview tasks

    Note: 1) __getitem__ method can return None, which means that requested
          index is not available (e.g. no person in the image). PyTorch native
          DataLoader doesn't support such behavior, so use safe
          dataset and dataloader from library 'nonechunks'
          (https://github.com/msamogh/nonechucks) which handles None elements.

          2) In DataLoader use custom collate_fn implemented as
          static method of this class
    """
    def __init__(self,
                 root,
                 image_shape=(256, 256),
                 kind='cmu',
                 ignore_list=[],
                 norm_image=True,
                 detection_conf_threshold=0.01,
                 scale_bbox=1.0,
                 frame_first=0,
                 frame_last=None,
                 skip_every_n_frames=1,
                 cuboid_side=200.0,
                 detector="ssd",
                 filter_keypoints_3d=False,
                 return_images=True,
                 crop_and_resize=True
                 ):
        self.root = os.path.normpath(root)
        self.scene_name = os.path.basename(self.root)

        self.image_shape = tuple(image_shape)
        self.kind = kind
        self.norm_image = norm_image
        self.detection_conf_threshold = detection_conf_threshold
        self.detector = detector
        self.scale_bbox = scale_bbox
        self.filter_keypoints_3d = filter_keypoints_3d

        self.frame_first = frame_first
        self.frame_last = frame_last
        self.skip_every_n_frames = skip_every_n_frames

        self.cuboid_side = cuboid_side

        self.return_images = return_images
        self.crop_and_resize = crop_and_resize

        self.ignore_list = ignore_list
        self.load_camera_names()
        self.load_cameras()

        self.load_image_dirs()
        self.check_consistency()

        self.load_detections()

        self.load_sample_list()

        if self.frame_last is None:
            total_n_frames = len(os.listdir(self.image_dirs[self.camera_names[0]]))
            self.frame_last = total_n_frames - 1

        self.__length = len(range(self.frame_first, self.frame_last + 1, self.skip_every_n_frames))

        # FIRE THESE LINES AFTER DEADLINE
        if self.scene_name in ('171204_pose5', '171204_pose6'):
            self.tri_pred_path = "/Vol1/dbstore/datasets/k.iskakov/logs_backup/multi-view-net/eval_tri_1stage_mrcnn_resnet_152_MVNetBasicResNet@15.03.2019-23:47:58/results__00000416.pkl"
        else:
            self.tri_pred_path = "/Vol1/dbstore/datasets/k.iskakov/logs_backup/multi-view-net/eval_tri_1stage_mrcnn_resnet_152_MVNetBasicResNet@15.03.2019-23:48:06/results__00001253.pkl"

        with open(self.tri_pred_path, 'rb') as fin:
            tri_pred = pickle.load(fin)
            stage_0 = np.concatenate(tri_pred['results']['stage_0'], axis=0)
            self.tri_pred = dict(zip(tri_pred['indexes'], stage_0))

        # if self.scene_name in ('171204_pose5', '171204_pose6'):
        #     with open("/Vol1/dbstore/datasets/k.iskakov/logs_backup/multi-view-net-supp/eval_cmu_tri_1stage_mrcnn_bboxes_mse_smooth_4_views_MVNetBasicResNet@27.03.2019-13:55:24/results__00000688.pkl", 'rb') as fin:
        #         tri_pred = pickle.load(fin)
        #     self.tri_pred = np.concatenate(tri_pred['results']['stage_0'], axis=0)
        # else:
        #     self.tri_pred = None


    def load_camera_names(self):
        self.camera_names = os.listdir(os.path.join(self.root, "hdImgs"))
        self.camera_names = list(filter(lambda x: x not in self.ignore_list, self.camera_names))
        self.camera_names = sorted(self.camera_names)


    def load_cameras(self):
        calibration_path = os.path.join(self.root, "calibration_{}.json".format(self.scene_name))
        with open(calibration_path) as fin:
            calibration_dict = json.load(fin)

        # camera params
        self.cameras = dict()
        for camera_rec in calibration_dict['cameras']:
            if camera_rec['name'] in self.camera_names:
                R = np.array(camera_rec['R'])
                t = np.array(camera_rec['t'])
                K = np.array(camera_rec['K'])
                dist = np.array(camera_rec['distCoef'])

                camera = Camera(R, t, K, dist, camera_rec['name'])
                self.cameras[camera_rec['name']] = camera

    def load_image_dirs(self):
        self.image_dirs = dict()
        camera_names_to_remove = []
        for camera_name in self.camera_names:
            image_dir = os.path.join(self.root, "hdImgs", camera_name)
            self.image_dirs[camera_name] = image_dir


    def load_detections(self):
        detections_dir = os.path.join(self.root, "{}-detections".format(self.detector))
        self.detections = dict()
        for camera_name in self.camera_names:
            with open(os.path.join(detections_dir, "{}.json".format(camera_name))) as fin:
                self.detections[camera_name] = json.load(fin)

    def check_consistency(self):
        # set of cameras cameras
        assert set(self.image_dirs.keys()) == set(self.cameras.keys()), "Different set of cameras in self.image_dirs and self.cameras"

        # check quantity of video frames
        length_set = set()
        for image_dir in self.image_dirs.values():
            length = len(os.listdir(image_dir))
            length_set.add(length)

        assert len(length_set) == 1, "Number of frames for different videos are not equal"

    def load_keypoints_3d(self, idx):
        hd_pose_json_path = os.path.join(self.root, "hdPose3d_stage1_coco19")
        hd_face_json_path = os.path.join(self.root, "hdFace3d")
        hd_hand_json_path = os.path.join(self.root, "hdHand3d")

        keypoints_3d = []

        # pose
        try:
            hd_pose_path = os.path.join(hd_pose_json_path, "body3DScene_{0:08d}.json".format(idx))
            with open(hd_pose_path) as fin:
                hd_pose = json.load(fin)

            keypoints_3d_pose = np.array(hd_pose['bodies'][0]['joints19']).reshape((-1, 4))
        except (FileNotFoundError, IndexError):
            keypoints_3d_pose = np.zeros((19, 4))

        if self.filter_keypoints_3d:
            if ('hd', "{0:08d}".format(idx)) not in self.sample_list[self.scene_name]:
                keypoints_3d_pose = np.zeros((19, 4))

        keypoints_3d.append(keypoints_3d_pose)

        # hands
        # hd_hands_path = os.path.join(hd_hand_json_path, "handRecon3D_hd{0:08d}.json".format(idx))
        # try:
        #     with open(hd_hands_path) as fin:
        #         hd_hands = json.load(fin)
        #     keypoints_3d_left_hand = np.hstack([np.array(hd_hands['people'][0]['left_hand']['landmarks']).reshape((-1, 3)),
        #                                         np.array(hd_hands['people'][0]['left_hand']['averageScore']).reshape((-1, 1))])
        #     keypoints_3d_right_hand = np.hstack([np.array(hd_hands['people'][0]['right_hand']['landmarks']).reshape((-1, 3)),
        #                                         np.array(hd_hands['people'][0]['right_hand']['averageScore']).reshape((-1, 1))])
        # except FileNotFoundError:
        #     keypoints_3d_left_hand = np.zeros((21, 4))
        #     keypoints_3d_right_hand = np.zeros((21, 4))
        #
        # keypoints_3d.extend([keypoints_3d_left_hand, keypoints_3d_right_hand])
        #
        # # face
        # try:
        #     hd_face_path = os.path.join(hd_face_json_path, "faceRecon3D_hd{0:08d}.json".format(idx))
        #     with open(hd_face_path) as fin:
        #         hd_face = json.load(fin)
        #         keypoints_3d_face = np.hstack([np.array(hd_face['people'][0]['face70']['landmarks']).reshape((-1, 3)),
        #                                        np.array(hd_face['people'][0]['face70']['averageScore']).reshape((-1, 1))])
        # except FileNotFoundError:
        #     keypoints_3d_face = np.zeros((70, 4))
        #
        # keypoints_3d.append(keypoints_3d_face)

        return np.vstack(keypoints_3d)

    def load_sample_list(self):
        path = os.path.join(self.root, "..", "sample_list.pkl")
        with open(path, 'rb') as fin:
            self.sample_list = pickle.load(fin)

    def __len__(self):
        return self.__length

    def __getitem__(self, idx, global_idx=None):
        idx = self.frame_first + idx * self.skip_every_n_frames

        sample = defaultdict(list)

        for camera_name in self.camera_names:
            if self.return_images:
                image_path = os.path.join(self.image_dirs[camera_name], "{:06}.jpg".format(idx))

                # load image
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # load detection
            try:
                detection = self.detections[camera_name][idx]
            except IndexError:
                print("Failed to take detection: idx={}, root={}".format(idx, self.root))
                exit()

            *bbox, c = detection
            if c < self.detection_conf_threshold:
                continue  # don't account this image

            bbox = tuple(map(int, bbox))
            camera = deepcopy(self.cameras[camera_name])
            if self.return_images:
                if self.crop_and_resize:
                    # crop and resize image
                    bbox = get_square_bbox(bbox)
                    bbox = scale_bbox(bbox, self.scale_bbox)

                    image = crop_image(image, bbox)
                    image_shape_before_resize = image.shape[:2]

                    image = resize_image(image, self.image_shape)

                    # update camera parameters because of crop and resize
                    camera.update_after_crop(bbox)
                    camera.update_after_resize(image_shape_before_resize, image.shape[:2])

                if self.norm_image:
                    image = normalize_image(image)

                sample['images'].append(image)
            else:
                camera = deepcopy(self.cameras[camera_name])

            sample['detections'].append(bbox + (c,))
            sample['cameras'].append(camera)

        # 3D keypoints
        keypoints_3d = self.load_keypoints_3d(idx)[:19]  # NOTE: for now return only body
        if self.kind == 'coco':
            keypoints_3d = cmu_to_coco(keypoints_3d)
        sample['keypoints_3d'] = keypoints_3d

        # build cuboid
        if self.kind == 'coco':
            base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
        elif self.kind == 'cmu':
            base_point = keypoints_3d[2, :3]

        sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        position = base_point - sides / 2
        sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # TODO: FIRE THESE LINES AFTER DEADLINE
        # print(self.tri_pred_path)

        sample['pred_keypoints_3d'] = self.tri_pred[global_idx]
        # sample['pred_keypoints_3d'] = sample['keypoints_3d'][:, :3]

        # print(global_idx)
        # print("diff", np.mean(np.abs(sample['pred_keypoints_3d'][11, :] - sample['keypoints_3d'][11, :3])))
        # print(sample['pred_keypoints_3d'][11, :], sample['keypoints_3d'][11, :3])
        sample['indexes'] = global_idx

        return sample


def cmu_to_coco(points):
    CMU_TO_COCO_MAP = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
    return points[CMU_TO_COCO_MAP]

def openpose_to_coco(points):
    OPENPOSE_TO_COCO_MAP = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    return points[OPENPOSE_TO_COCO_MAP]
