# -*- coding: utf-8 -*-
# Author: Sizhe Wei <sizhewei@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Date: 2023-04-10

'''
Use where2comm as backbone
We just need to learn the flow generator for each cav
Training part is MotionNet
'''
from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.raindrop_flow_regular import raindrop_fuse
import torch

class PointPillarWhere2commFlowRegular(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commFlowRegular, self).__init__()

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

        self.compensation = False
        self.rain_fusion = raindrop_fuse(args['rain_model'])

        self.multi_scale = args['rain_model']['multi_scale']

        self.single_supervise = False
        
        if self.shrink_flag:
            dim = args['shrink_header']['dim'][0]
            self.cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)


        self.backbone_fix_flag = False
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix_flag = True
            self.backbone_fix()
            print('=== backbone fixed ===')

        self.only_tune_header_flag = False
        if 'only_tune_header' in args.keys() and args['only_tune_header']:
            self.only_tune_header_flag = True
            self.only_tune_header()
            print('=== only tune header ===')

        assert self.backbone_fix_flag == False or self.only_tune_header_flag == False, 'backbone_fix and only_tune_header cannot be True at the same time'
    
    def only_tune_header(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.rain_fusion.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
    
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
        pastk_2_past0_tr_mats = data_dict['pastk_2_past0_tr_mats']              #(sum(N_b), k, 4, 4)
        
        # debug = 0
        B, _, k, _, _ = pairwise_t_matrix.shape
        # for i in range(B):
        #     debug += record_len[i]*k

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c  ('pillar_features')
        batch_dict = self.pillar_vfe(batch_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        # import ipdb; ipdb.set_trace()
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # debug use
        flow_gt = data_dict['label_dict']['flow_gt']
        warp_mask = data_dict['label_dict']['warp_mask']

        # rain attention:
        if self.multi_scale:
            fused_feature, communication_rates, result_dict, flow, state_preds = self.rain_fusion(batch_dict['spatial_features'],
                psm_single,
                record_len,
                pairwise_t_matrix, 
                record_frames,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
                pastk_2_past0_tr_mats=pastk_2_past0_tr_mats,
                flow_gt=flow_gt, 
                warp_mask=warp_mask)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
                if self.single_supervise:
                    single_feature = self.shrink_conv(single_feature)
        else:
            fused_feature, communication_rates, result_dict = self.rain_fusion(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix,
                                            record_frames)
            if self.compensation:
                if self.single_supervise:
                    fused_feature, single_feature, communication_rates, all_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                else:
                    fused_feature,fused_feature_curr,fused_feature_latency, communication_rates, all_recon_loss, all_latency_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])            
        
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        # fuse 之后的 feature (ego)
        output_dict = {'psm': psm,
                       'rm': rm}

        output_dict.update({'flow_preds': flow,
                            'state_preds': state_preds})

        if self.compensation:
            if self.single_supervise:
                psm_nonego_single = self.cls_head(single_feature)
                rm_nonego_single = self.reg_head(single_feature)
                output_dict.update({
                    'psm_nonego_single': psm_nonego_single,
                    'rm_nonego_single': rm_nonego_single
                })
            
            output_dict.update({
                'recon_loss': all_recon_loss, 
                'record_len': record_len
            })

        output_dict.update({'psm_single': psm_single,
                       'rm_single': rm_single,
                       'comm_rate': communication_rates
                       })
        
        output_dict.update(result_dict) 
        
        return output_dict
