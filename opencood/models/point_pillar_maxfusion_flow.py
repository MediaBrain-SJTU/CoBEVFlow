# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from opencood.models.fuse_modules.max_fuse import MaxFusion

class PointPillarMaxfusionFlow(nn.Module):
    def __init__(self, args):
        super(PointPillarMaxfusionFlow, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], # 384
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], # 384
                                  kernel_size=1)

        self.fusion_net = MaxFusion(args['fusion_args'])

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False

    def forward(self, data_dict, stage = 1, pairwise_t_matrix=None):
        if stage == 1:
            return self.single_forward(data_dict)
        else:
            return self.fuse_forward(data_dict, pairwise_t_matrix)
        
    def single_forward(self, data_dict):    
        output_dict = {}
        intermediate_feature_flow = True

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']  # B, C, H, W

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if intermediate_feature_flow:
            output_dict.update({
                'spatial_features_2d': spatial_features_2d  # B, C, H, W
            })
        
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict.update({
            'psm': psm,
            'rm': rm
        })
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dm': dm})

        return output_dict

    def fuse_forward(self, updated_dict, pairwise_t_matrix):
        """ 参考 where2comm 的写法 如何将多车的feature 融合起来 返回最终的output dict
        Parameters:
        -----------
        updated_dict : 

        Returns:
        ---------
        output_dict : 

        """
        device = updated_dict['ego']['spatial_features_2d'].device
        spatial_feature_2d_list = []
        for cav_id, cav_content in updated_dict.items():
            spatial_feature_2d_list.append(cav_content['updated_spatial_feature_2d'])

        record_len = torch.tensor([len(spatial_feature_2d_list)]).to(device)
        spatial_feature_2d = torch.stack(spatial_feature_2d_list, dim=0).to(device)  # (sum(cav), C, H, W)

        pairwise_t_matrix = pairwise_t_matrix.unsqueeze(0)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        fused_feature = self.fusion_net(spatial_feature_2d,
                                        record_len,
                                        pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict