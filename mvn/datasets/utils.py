import numpy as np
import torch

from mvn.utils.img import image_batch_to_torch

def make_collate_fn(randomize_n_views=True, min_n_views=1, max_n_views=4):
    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        batch = dict()

        if len(items) == 0:
            print("All items in batch are None")
            return None

        total_n_views = min(len(item['images']) for item in items)

        if total_n_views < 2 and randomize_n_views:
            print("total_n_views < 2 for randomize_n_views!")

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

        if type(items[0]['keypoints_3d']) is list:
            batch['keypoints_3d'] = np.stack([np.stack([item['keypoints_3d'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        else:
            batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]

        if type(items[0]['indexes']) is list:
            batch['indexes'] = np.stack([item['indexes'] for item in items], axis=0)
        else:
            batch['indexes'] = [item['indexes'] for item in items]

        batch['images_paths'] = [item['images_paths'] for item in items]

        return batch
    return collate_fn

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_batch(batch, device):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        # image_batch /= 255.0
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[..., :3]).float().to(device)

    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[..., 3:]).float().to(device)

    # projection matricies
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)

    return images_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, proj_matricies_batch
