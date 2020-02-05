import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvn.utils.img import to_numpy, to_torch
from mvn.utils import multiview, volumetric
from IPython.core.debugger import set_trace



def get_coord_volumes(kind, 
                        training, 
                        rotation,
                        cuboid_side, 
                        volume_size, 
                        device, 
                        keypoints = None,
                        batch_size = None,
                        dt = None
                        ):
    
        use_default_basepoint = keypoints is None
        bs_dt = (batch_size, dt) if use_default_basepoint else keypoints.shape[:-2]
        sides = torch.tensor([cuboid_side, cuboid_side, cuboid_side], dtype=torch.float).to(device)

        # default base_points are the coordinate's origins
        base_points = torch.zeros((*bs_dt, 3), dtype=torch.float).to(device)
        
        if not use_default_basepoint:    
            # get root (pelvis) from keypoints   
            if kind == "coco":
                base_points = (keypoints[...,11, :3] + keypoints[...,12, :3]) / 2
            elif kind == "mpii":
                base_points = keypoints[..., 6, :3] 

        position = base_points - sides / 2

        # build cuboids
        cuboids = None

        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size, device=device),
                                        torch.arange(volume_size, device=device),
                                         torch.arange(volume_size, device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.view((-1, 3))
        grid = grid.view(*[1]*len(bs_dt), *grid.shape).repeat(*keypoints.shape[:-2], *[1]*len(grid.shape))

        grid[..., 0] = position[..., 0].unsqueeze(-1) + (sides[0] / (volume_size - 1)) * grid[..., 0]
        grid[..., 1] = position[..., 1].unsqueeze(-1) + (sides[1] / (volume_size - 1)) * grid[..., 1]
        grid[..., 2] = position[..., 2].unsqueeze(-1) + (sides[2] / (volume_size - 1)) * grid[..., 2]
        
        if kind == "coco":
            axis = [0, 1, 0]  # y axis
        elif kind == "mpii":
            axis = [0, 0, 1]  # z axis
            
        # random rotation
        if training and rotation:    
            
            center = torch.tensor(base_points).type(torch.float).to(device).unsqueeze(-2)
            grid = grid - center
            grid = torch.stack([volumetric.rotate_coord_volume(coord_grid,\
                                np.random.uniform(0.0, 2 * np.pi), axis) for coord_grid in grid])
            grid = grid + center

        grid = grid.view(*bs_dt, volume_size, volume_size, volume_size, 3)
    
        return grid, cuboids, base_points


def root_centering(keypoints, kind, inverse = False):

    '''
    Makes other keypoints to become root relative, undoes it if inverse = True
    '''    
    if kind == "human36m":
        base_joint = 6
    elif kind == "coco":
        base_joint = 11

    n_joints = keypoints.shape[-2]

    keypoints_transformed = keypoints.clone()
    if inverse:
        keypoints_transformed[..., torch.arange(n_joints) != base_joint,:] += keypoints_transformed[..., base_joint:base_joint + 1,:]
    else:
        keypoints_transformed[..., torch.arange(n_joints) != base_joint,:] -= keypoints_transformed[..., base_joint:base_joint + 1,:]

    return keypoints_transformed


def keypoints_to_features(keypoints):
    
    bs = keypoints.shape[0]
    features=[]
    for i in range(bs):
        features.append(torch.cat([keypoints[0].view(-1), F.pdist(keypoints[i])]))

    return torch.stack(features,0)


def integrate_tensor_2d(heatmaps, softmax=True):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps


def integrate_tensor_3d(volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape
    volumes = volumes + torch.rand_like(volumes) * 1e-4
    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))

    mass_x = volumes.sum(dim=3).sum(dim=3)
    mass_y = volumes.sum(dim=2).sum(dim=3)
    mass_z = volumes.sum(dim=2).sum(dim=2)

    mass_times_coord_x = mass_x * torch.arange(x_size).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(y_size).type(torch.float).to(mass_y.device)
    mass_times_coord_z = mass_z * torch.arange(z_size).type(torch.float).to(mass_z.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)
    z = mass_times_coord_z.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)
        z = z / mass_z.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y, z), dim=2)
    coordinates = coordinates.reshape((batch_size, n_volumes, 3))

    return coordinates, volumes


def integrate_tensor_3d_with_coordinates(volumes, coord_volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

    return coordinates, volumes


def unproject_heatmaps(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method='sum', vol_confidences=None, volumes_multipliers=None):
    device = heatmaps.device
    batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
    volume_shape = coord_volumes.shape[1:4]

    volume_batch = [] if volume_aggregation_method=='no_aggregation' else torch.zeros(batch_size, n_joints, *volume_shape, device=device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        coord_volume = coord_volumes[batch_i]
        grid_coord = coord_volume.reshape((-1, 3))

        volume_batch_to_aggregate = torch.zeros(n_views, n_joints, *volume_shape, device=device)

        for view_i in range(n_views):
            heatmap = heatmaps[batch_i, view_i]
            heatmap = heatmap.unsqueeze(0)

            grid_coord_proj = multiview.project_3d_points_to_image_plane_without_distortion(proj_matricies[batch_i, view_i],
                                                                                            grid_coord, 
                                                                                            convert_back_to_euclidean=False
                                                                                            )

            invalid_mask = grid_coord_proj[:, 2] <= 0.0  # depth must be larger than 0.0

            grid_coord_proj[grid_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
            grid_coord_proj = multiview.homogeneous_to_euclidean(grid_coord_proj)

            # transform to [-1.0, 1.0] range
            grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
            grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / heatmap_shape[0] - 0.5)
            grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / heatmap_shape[1] - 0.5)
            grid_coord_proj = grid_coord_proj_transformed # [N,2]
            # prepare to F.grid_sample
            grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)
            current_volume = F.grid_sample(heatmap, grid_coord_proj)

            # zero out non-valid points
            current_volume = current_volume.view(n_joints, -1)
            current_volume[:, invalid_mask] = 0.0

            # reshape back to volume
            current_volume = current_volume.view(n_joints, *volume_shape)

            # collect
            volume_batch_to_aggregate[view_i] = current_volume

        # agregate resulting volume
        if volume_aggregation_method.startswith('conf'):
            volume_batch[batch_i] = (volume_batch_to_aggregate * vol_confidences[batch_i].view(n_views, n_joints, 1, 1, 1)).sum(0)
        elif volume_aggregation_method == 'sum':
            volume_batch[batch_i] = volume_batch_to_aggregate.sum(0)
        elif volume_aggregation_method == 'max':
            volume_batch[batch_i] = volume_batch_to_aggregate.max(0)[0]
        elif volume_aggregation_method == 'softmax':
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone()
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, -1)
            volume_batch_to_aggregate_softmin = nn.functional.softmax(volume_batch_to_aggregate_softmin, dim=0)
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, n_joints, *volume_shape)

            volume_batch[batch_i] = (volume_batch_to_aggregate * volume_batch_to_aggregate_softmin).sum(0)

        elif volume_aggregation_method == 'no_aggregation':
            if volumes_multipliers is not None:
                volume_batch.append(volume_batch_to_aggregate*volumes_multipliers[view_i])
            else:
                volume_batch.append(volume_batch_to_aggregate)
                    
        else:
            raise ValueError("Unknown volume_aggregation_method: {}".format(volume_aggregation_method))

    return volume_batch


def gaussian_2d_pdf(coords, means, sigmas, normalize=True):
    normalization = 1.0
    if normalize:
        normalization = (2 * np.pi * sigmas[:, 0] * sigmas[:, 0])

    exp = torch.exp(-((coords[:, 0] - means[:, 0]) ** 2 / sigmas[:, 0] ** 2 + (coords[:, 1] - means[:, 1]) ** 2 / sigmas[:, 1] ** 2) / 2)
    return exp / normalization


def render_points_as_2d_gaussians(points, sigmas, image_shape, normalize=True):
    device = points.device
    n_points = points.shape[0]

    yy, xx = torch.meshgrid(torch.arange(image_shape[0]).to(device), torch.arange(image_shape[1]).to(device))
    grid = torch.stack([xx, yy], dim=-1).type(torch.float32)
    grid = grid.unsqueeze(0).repeat(n_points, 1, 1, 1)  # (n_points, h, w, 2)
    grid = grid.reshape((-1, 2))

    points = points.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0], image_shape[1], 1)
    points = points.reshape(-1, 2)

    sigmas = sigmas.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0], image_shape[1], 1)
    sigmas = sigmas.reshape(-1, 2)

    images = gaussian_2d_pdf(grid, points, sigmas, normalize=normalize)
    images = images.reshape(n_points, *image_shape)

    return images
