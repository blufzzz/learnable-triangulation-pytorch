import numpy as np
import torch
import os
from IPython.core.debugger import set_trace
from mvn.utils.img import image_batch_to_torch

def make_collate_fn(randomize_n_views=True, min_n_views=1, max_n_views=4, only_keypoints=False):
    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        batch = dict()

        if len(items) == 0:
            # print("All items in batch are None")
            return None

        total_n_views = min(len(item['images']) for item in items)

        if total_n_views < 2 and randomize_n_views:
            print("total_n_views < 2 for randomize_n_views!")
            raise RuntimeError

        if min_n_views is not None and total_n_views < min_n_views:
            print("total_n_views < min_n_views! Returns `None` batch")
            return None

        if max_n_views is not None and total_n_views > max_n_views:
            print("total_n_views > max_n_views! Returns `None` batch")
            return None     

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)
        
        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        # batch['image_shapes_before_resize'] = np.array([[item['image_shapes_before_resize'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]
        batch['images_paths'] = [item['images_paths'] for item in items]

        if type(items[0]['keypoints_3d']) is list:
            batch['keypoints_3d'] = np.stack([np.stack([item['keypoints_3d'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        else:
            batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]

        if type(items[0]['indexes']) is list:
            batch['indexes'] = np.stack([item['indexes'] for item in items], axis=0)
        else:
            batch['indexes'] = [item['indexes'] for item in items]

        return batch

    def collate_fn_keypoints(items):
        items = list(filter(lambda x: x is not None, items))
        batch = dict()
        if len(items) == 0:
            # print("All items in batch are None")
            return None
        if type(items[0]['keypoints_3d']) is list:
            batch['keypoints_3d'] = np.stack([np.stack([keypoints for keypoints in item['keypoints_3d']],0) for item in items],0)
        else:
            batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]

        if type(items[0]['indexes']) is list:
            batch['indexes'] = np.stack([item['indexes'] for item in items], axis=0)
        else:
            batch['indexes'] = [item['indexes'] for item in items]

        return batch
    
    if only_keypoints:
        return collate_fn_keypoints
    else:
        return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_batch(batch, device, only_keypoints=False):
    
    if not only_keypoints:
        # images
        images_batch = []
        for image_batch in batch['images']:
            image_batch = image_batch_to_torch(image_batch)
            image_batch = image_batch.to(device)
            images_batch.append(image_batch)

        images_batch = torch.stack(images_batch, dim=0)

        # projection matricies
        proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in \
                                 camera_batch], dim=0) for camera_batch in \
                                    batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        
        proj_matricies_batch = proj_matricies_batch.float().to(device)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[..., :3]).float().to(device)

    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[..., 3:]).float().to(device)

    if not only_keypoints:
        output = (images_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, proj_matricies_batch)
    else:
        output = (keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt)

    return output



# Loading utilities
def load_objects(obj_root):
    object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),
            'faces': np.array(mesh.faces)
        }
    return all_models


def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)[sample['frame_idx']]
    return skeleton


def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                            sample['seq_idx'], 'object_pose.txt')
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample['frame_idx']]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    print('Loading obj transform from {}'.format(seq_path))
    return trans_matrix


# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)