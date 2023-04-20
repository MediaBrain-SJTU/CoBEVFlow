# -*- coding: utf-8 -*-
# Author: Sizhe Wei <sizhewei@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
'''
Use where2comm as backbone
First generate the box
then use the box to generate the flow
利用box生成flow
将该flow通过某种方法warp feature 
将补偿好的feature融合 得到最终结果
'''

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
# from opencood.models.fuse_modules.raindrop_attn import raindrop_fuse
from opencood.models.fuse_modules.raindrop_swin import raindrop_swin
from opencood.models.fuse_modules.raindrop_swin_w_single import raindrop_swin_w_single
from opencood.models.fuse_modules.raindrop_flow import raindrop_fuse
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np

class PointPillarWhere2commFlowTwoStage(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commFlowTwoStage, self).__init__()

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

        if 'design_mode' in args.keys():
            self.design_mode = args['design_mode']
            print(f'=== design mode : {self.design_mode} ===')

        # 用于 single 部分的监督
        self.single_supervise = False
        if 'with_compensation' in args and args['with_compensation']:
            self.compensation = True
            if 'with_single_supervise' in args and args['with_single_supervise']:
                self.rain_fusion = raindrop_swin_w_single(args['rain_model'])
                self.single_supervise = True
            else:
                self.rain_fusion = raindrop_swin(args['rain_model'])
        else: 
            self.compensation = False
            self.rain_fusion = raindrop_fuse(args['rain_model'], self.design_mode)

        self.multi_scale = args['rain_model']['multi_scale']
        
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

        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
            print('=== backbone fixed ===')

        self.matcher = Matcher()
    
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

    def generate_box_flow(self, data_dict, pred_dict, dataset, device): 
        """
        data_dict : 

        pred_dict : 
            {
                psm_single_list: len = B, each element's shape is (N_b x k, 2, H, W) 
                rm_single_list: len = B, each element's shape is (N_b x k, 14, H, W) 
            }
        """
        # for b in range(B):
        # 1. get box results
        lidar_pose_batch = self.regroup(data_dict['past_lidar_pose'], data_dict['record_len'], k=1)
        past_k_time_diff = self.regroup(data_dict['past_k_time_interval'], data_dict['record_len'], k=3)
        anchor_box = data_dict['anchor_box'] # (H, W, 2, 7)
        psm_single_list = pred_dict['psm_single_list']
        rm_single_list = pred_dict['rm_single_list']
        
        trans_mat_pastk_2_past0_batch = []
        B = len(lidar_pose_batch)
        box_flow_map_list = []
        reserved_mask_list = []
        # for all batches
        for b in range(B):
            psm_single = psm_single_list[b].reshape(-1, self.k, 2, psm_single_list[b].shape[-2], psm_single_list[b].shape[-1]) # (N_b, k, 2, H, W)
            rm_single = rm_single_list[b].reshape(-1, self.k, 14, rm_single_list[b].shape[-2], rm_single_list[b].shape[-1]) # (N_b, k, 14, H, W)
            cav_past_k_time_diff = past_k_time_diff[b]
            cav_trans_mat_pastk_2_past0 = []
            # for all cavs
            '''
            box_result : dict for each cav at each time
            {
                cav_idx : {
                    'past_k_time_diff' : 
                    [0] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                        pred_box_center_tensor : (n, 7)
                        scores: (n, )
                    },
                    ...
                    [k-1] : { ... }
                }
            }
            '''
            box_results = OrderedDict()
            for cav_idx in range(data_dict['record_len'][b]):
                # generate one cav's trans_mat_pastk_2_past0
                pastk_trans_mat_pastk_2_past0 = []
                for i in range(self.k):
                    unit_mat = x1_to_x2(lidar_pose_batch[b][cav_idx, i, :].cpu().numpy(), lidar_pose_batch[b][cav_idx, 0].cpu()) # (4, 4)
                    pastk_trans_mat_pastk_2_past0.append(unit_mat)
                pastk_trans_mat_pastk_2_past0 = torch.from_numpy(np.stack(pastk_trans_mat_pastk_2_past0, axis=0)).to(device) # (k, 4, 4)
                
                # 1. generate one cav's box results
                box_results[cav_idx] = dataset.generate_pred_bbx_frames(psm_single[cav_idx], rm_single[cav_idx], pastk_trans_mat_pastk_2_past0, cav_past_k_time_diff[cav_idx*self.k:cav_idx*self.k+self.k], anchor_box)

            cav_trans_mat_pastk_2_past0.append(pastk_trans_mat_pastk_2_past0)
            
            # 2. generate box flow in one batch
            box_flow_map, mask = self.matcher(box_results, fusion='flow', shape_list=torch.tensor([64, 200, 704]).to(device))
            box_flow_map_list.append(box_flow_map)
            reserved_mask_list.append(mask)
        
        final_flow_map = torch.concat(box_flow_map_list, dim=0)
        final_reserved_mask = torch.concat(reserved_mask_list, dim=0)
        return final_flow_map, final_reserved_mask

    def forward(self, data_dict, dataset=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']
        record_frames = data_dict['past_k_time_interval']                       #(B, )
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4)
        
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

        ####### debug use, viz feature of each cav
        # from matplotlib import pyplot as plt
        # viz_save_path = '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/debug_4_feature_flow/where2comm'
        # for i in range(spatial_features_2d.shape[0]):
        #     viz_content = torch.max(spatial_features_2d[i], dim=0)[0].detach().cpu()
        #     plt.imshow(viz_content)
        #     plt.savefig(viz_save_path+f'/updated_feature_{i}.png')
        ##############
        
        # dcn
        # if self.dcn:
        #     spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # generate box flow
        # [B, 256, 50, 176]
        single_output = {}
        single_output.update({'psm_single_list': self.regroup(psm_single, record_len, k), 
        'rm_single_list': self.regroup(rm_single, record_len, k)})
        box_flow_map, reserved_mask = self.generate_box_flow(data_dict, single_output, dataset, psm_single.device)

        flow_gt = data_dict['label_dict']['flow_gt']

        # rain attention:
        if self.multi_scale:
            fused_feature, communication_rates, result_dict, flow_recon_loss = self.rain_fusion(batch_dict['spatial_features'],
                psm_single,
                record_len,
                pairwise_t_matrix, 
                record_frames,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
                box_flow=box_flow_map, reserved_mask=reserved_mask,
                flow_gt=flow_gt)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
                if self.single_supervise:
                    single_feature = self.shrink_conv(single_feature)
                # if self.compensation:
                    
                #         # single_feature = self.shrink_conv(single_feature)
                
                #         fused_feature_curr = self.shrink_conv(fused_feature_curr)
                #         fused_feature_latency = self.shrink_conv(fused_feature_latency)
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

        ####### debug use, viz fused feature of where2comm
        # viz_content = torch.max(fused_feature[0], dim=0)[0].detach().cpu()
        # plt.imshow(viz_content)
        # plt.savefig(viz_save_path+'/fused_feature.png')  
        ##############
        
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        # fuse 之后的 feature (ego)
        output_dict = {'psm': psm,
                       'rm': rm}

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

        output_dict.update({'flow_recon_loss': flow_recon_loss})
        
        output_dict.update(result_dict) 
        
        return output_dict
