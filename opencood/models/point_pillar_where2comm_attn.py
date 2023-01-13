# -*- coding: utf-8 -*-
# Author: Sizhe Wei <sizhewei@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.raindrop_attn import raindrop_fuse
import torch

class PointPillarWhere2commAttn(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commAttn, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'num_sweep_frames' in args:    # number of frames we use in LSTM
            self.k = args['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in args:          # number of time delay
            self.tau = args['time_delay'] 
        else:
            self.tau = 0

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        self.rain_fusion = raindrop_fuse(args['rain_model'])
        self.multi_scale = args['rain_model']['multi_scale']
        
        if self.shrink_flag:
            self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len, k=1):
        cum_sum_len = torch.cumsum(record_len*k, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']
        record_frames = data_dict['past_k_time_interval']                       #(B, )
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c  ('pillar_features')
        batch_dict = self.pillar_vfe(batch_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 252]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        # if self.dcn:
        #     spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # rain attention:
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'],
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix,
                                            record_frames)

        # # print('spatial_features_2d: ', spatial_features_2d.shape)
        # if self.multi_scale:
        #     fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
        #                                     psm_single,
        #                                     record_len,
        #                                     pairwise_t_matrix, 
        #                                     self.backbone,
        #                                     [self.shrink_conv, self.cls_head, self.reg_head])
        #     # downsample feature to reduce memory
        #     if self.shrink_flag:
        #         fused_feature = self.shrink_conv(fused_feature)
        # else:
        #     fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
        #                                     psm_single,
        #                                     record_len,
        #                                     pairwise_t_matrix)
            
            
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm
                       }
        output_dict.update(result_dict)
        
        voxel_features = data_dict['curr_processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['curr_processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['curr_processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']                  
        
        batch_single_dict = {'voxel_features': voxel_features,
                            'voxel_coords': voxel_coords,
                            'voxel_num_points': voxel_num_points,
                            'record_len': record_len}
        # n, 4 -> n, c  ('pillar_features')
        batch_single_dict = self.pillar_vfe(batch_single_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        batch_single_dict = self.scatter(batch_single_dict)
        batch_single_dict = self.backbone(batch_single_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 252]
        single_spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            single_spatial_features_2d = self.shrink_conv(single_spatial_features_2d)  # (B, 256, H', W')
        # compressor
        if self.compression:
            single_spatial_features_2d = self.naive_compressor(single_spatial_features_2d)
        # [B, 256, 50, 176]
        psm_single = self.cls_head(single_spatial_features_2d)
        rm_single = self.reg_head(single_spatial_features_2d)

        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict
