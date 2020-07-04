import numpy as np
import pickle
import random
from collections import defaultdict
import sys
import torch
from torch import nn
import torch.nn.functional as F
import os


from mvn.models import pose_resnet, pose_hrnet
from time import time
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, R2D, SqueezeLayer
from mvn.models.v2v_models import V2VModel_v1
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                get_encoder, \
                                FeatureDecoderLSTM, \
                                StylePosesLSTM

from IPython.core.debugger import set_trace

STYLE_VECTOR_CONST = None


class VolumetricSpadeDebug(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.keypoints_per_frame = config.dataset.keypoints_per_frame if \
                                         hasattr(config.dataset, 'keypoints_per_frame') else False

        # temporal 
        self.dt  = config.dataset.dt
        self.pivot_type = config.dataset.pivot_type
        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[self.pivot_type]
        self.aux_indexes = list(range(self.dt))
        self.aux_indexes.remove(self.pivot_index)

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # cuboid
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier
        self.rotation = config.model.rotation

        # pelvis
        self.pelvis_type = config.model.pelvis_type if hasattr(config.model, 'pelvis_type') else 'gt'

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, 'transfer_cmu_to_human36m') else False

        # modules params
        self.temporal_condition_type = config.model.temporal_condition_type

        self.f2v_type = config.model.f2v_type
        self.f2v_normalization_type = config.model.f2v_normalization_type
        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        assert self.v2v_normalization_type in ['group_norm','batch_norm']
        self.include_pivot = config.model.include_pivot

        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim

        ############
        # BACKBONE #   
        ############
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        #######
        # V2V #
        #######
        v2v_input_features_dim = (self.volume_features_dim + self.num_joints) if \
                                  self.temporal_condition_type in ['stack', 'stack_poses'] else self.volume_features_dim
        assert self.v2v_type == 'conf'
        use_compound_norm = config.model.use_compound_norm if hasattr(config.model, 'use_compound_norm') else True
        temporal_condition_type = None if self.temporal_condition_type == 'stack' else self.temporal_condition_type
        self.volume_net = V2VModel(v2v_input_features_dim,
                                   self.num_joints,
                                   v2v_normalization_type=self.v2v_normalization_type,
                                   config=config.model.v2v_configuration,
                                   style_vector_dim=self.style_vector_dim,
                                   params_evolution=False,
                                   style_forward=False,
                                   use_compound_norm=use_compound_norm,
                                   temporal_condition_type=temporal_condition_type)
        ####################
        # process_features #
        ####################
        if self.volume_features_dim != 256:    
            self.process_features = nn.Sequential(nn.Conv2d(256, self.volume_features_dim, 1))
        else:
            self.process_features = nn.Sequential()   

        #######
        # F2V #
        #######
        if self.f2v_type != 'target_heatmaps':
            assert self.f2v_type == 'v2v'
            self.features_sequence_to_vector = V2VModel(self.volume_features_dim,
                                                       self.num_joints,
                                                       v2v_normalization_type=self.f2v_normalization_type,
                                                       config=config.model.f2v_configuration,
                                                       style_vector_dim=None,
                                                       params_evolution=False,
                                                       style_forward=False,
                                                       use_compound_norm=False,
                                                       temporal_condition_type=None)


            # load v2v
            state_dict = torch.load(config.model.baseline_checkpoint)['model_state']
            model_state_dict = self.features_sequence_to_vector.state_dict() 
            prefix = 'volume_net.'
            for k,v in state_dict.items():
                if prefix in k:
                    k = k.replace(prefix, "")
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                    else:
                        set_trace()
            self.features_sequence_to_vector.load_state_dict(model_state_dict, strict=True)
            del model_state_dict 
            torch.cuda.empty_cache()
            print("LOADED PRE-TRAINED features_sequence_to_vector!!!")

            # load process features
            model_state_dict = self.process_features.state_dict() 
            prefix = 'process_features.'
            for k,v in state_dict.items():
                if prefix in k:
                    k = k.replace(prefix, "")
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                    else:
                        set_trace()
            self.process_features.load_state_dict(model_state_dict, strict=True)
            del model_state_dict 
            torch.cuda.empty_cache()
            print("LOADED PRE-TRAINED process_features!!!")

            # load process features
            model_state_dict = self.backbone.state_dict() 
            prefix = 'backbone.'
            for k,v in state_dict.items():
                if prefix in k:
                    k = k.replace(prefix, "")
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                    else:
                        set_trace()
            self.backbone.load_state_dict(model_state_dict, strict=True)
            del state_dict
            del model_state_dict 
            torch.cuda.empty_cache()
            print("LOADED PRE-TRAINED backbone!!!")

        description(self)


    def forward(self, 
                images_batch, 
                batch, 
                randomize_style=False, 
                const_style_vector=False,
                return_me_vector = False,
                debug=False,
                master=True):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt
        ######################
        # FEATURE ECTRACTION #   
        ######################
        # forward backbone
        grad_context = torch.autograd.enable_grad if self.f2v_type == 'target_heatmaps' else torch.no_grad
        with grad_context():
            heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch.view(-1, 3, *image_shape))

        # extract aux_features
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]

        pivot_features = features.view(batch_size, features_channels, *features_shape)
        pivot_features = self.process_features(pivot_features).unsqueeze(1)
            
        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch[:,self.pivot_index,...].unsqueeze(1) # pivot camera 

        ##########
        # PELVIS #   
        ##########            
        if self.pelvis_type =='gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
        #####################
        # VOLUMES CREATING  #   
        #####################
        coord_volumes, _, base_points = get_coord_volumes(self.kind, 
                                                            self.training, 
                                                            self.rotation,
                                                            self.cuboid_side,
                                                            self.volume_size, 
                                                            device,
                                                            keypoints=tri_keypoints_3d
                                                            )

        ###############
        # V2V FORWARD #   
        ###############
        unproj_features = unproject_heatmaps(pivot_features,  
                                            proj_matricies_batch,
                                            coord_volumes, 
                                            volume_aggregation_method=self.volume_aggregation_method,
                                            vol_confidences=vol_confidences
                                            )

        if self.f2v_type == 'v2v':
            with torch.no_grad():  
                style_vector = self.features_sequence_to_vector(unproj_features) # [batch_size, style_vector_dim]

        #########################
        # TARGET HEATMAPS STYLE #
        #########################
        elif self.f2v_type == 'target_heatmaps':
            coord_volume_unsq = coord_volumes.unsqueeze(1)
            keypoints_gt_i_unsq = tri_keypoints_3d.unsqueeze(2).unsqueeze(2).unsqueeze(2)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            H = torch.zeros_like(dists)

            for k,w in zip([1,2,3,4,5],
                           [1,1,1,1,1]):
                n_cells = k**3
                knn = dists.view(*dists.shape[:-3],-1).topk(n_cells,dim=-1,largest=False)
                radius = knn.values.max(dim=-1)[0].unsqueeze(2).unsqueeze(2).unsqueeze(2)
                mask = dists <= radius
                # try:
                #     assert mask.sum() / n_cells == 17
                # except Exception as e:
                #     set_trace()
                H[mask] += w   
            style_vector = F.softmax(H.view(*H.shape[:-3], -1),dim=-1).view(*mask.shape) 
        else:
            raise RuntimeError('No temporal feature extractor has been defined!')
        
        style_vector_volumes = F.interpolate(style_vector,
                                             size=(self.volume_size,self.volume_size,self.volume_size), 
                                             mode='trilinear')
        ########################## 
        # VOLUMES FEEDING TO V2V #   
        ##########################
        torch.cuda.empty_cache()         
        if self.temporal_condition_type == 'spade':
            volumes = self.volume_net(unproj_features, params=style_vector_volumes)
        elif self.temporal_condition_type == 'stack':
            volumes = self.volume_net(torch.cat([unproj_features, style_vector_volumes], 1))
        else:
            raise RuntimeError('Wrong self.temporal_condition_type, should be in [`stack`, `spade`]')    
                
        # integral 3d
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                         coord_volumes,
                                                                         softmax=self.volume_softmax)

        return [vol_keypoints_3d,
                volumes,
                None, # confidences
                None, # cuboids
                coord_volumes,
                base_points
                ]














