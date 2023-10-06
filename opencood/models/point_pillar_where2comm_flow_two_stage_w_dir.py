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
# from opencood.models.sub_modules.max_resnet_bev_backbone import MaxResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.raindrop_swin import raindrop_swin
from opencood.models.fuse_modules.raindrop_swin_w_single import raindrop_swin_w_single
from opencood.models.fuse_modules.raindrop_flow import raindrop_fuse
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.

    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree

    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    return pose_noise

def get_past_k_pairwise_transformation2ego(past_k_lidar_pose, noise_level, k=3, max_cav=5):
    """
    Get transformation matrixes accross different agents to curr ego at all past timestamps.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.
    
    ego_pose : list
        ego pose

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The transformation matrix each cav to curr ego at past k frames.
        shape: (L, k, 4, 4), L is the max cav number in a scene, k is the num of past frames
        pairwise_t_matrix[i, j] is T i_to_ego at past_j frame
    """
    pos_std = noise_level['pos_std']
    rot_std = noise_level['rot_std']
    pos_mean = 0 
    rot_mean = 0
    
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, k, 1, 1)) # (L, k, 4, 4)

    ego_pose = past_k_lidar_pose[0, 0]

    t_list = []

    # save all transformation matrix in a list in order first.
    for cav_id in range(past_k_lidar_pose.shape[0]):
        past_k_poses = []
        for time_id in range(k):
            loc_noise = generate_noise(pos_std, rot_std)
            past_k_poses.append(x_to_world(past_k_lidar_pose[cav_id, time_id].cpu().numpy()+loc_noise))
        t_list.append(past_k_poses) # Twx
    
    ego_pose = x_to_world(ego_pose.cpu().numpy())
    for i in range(len(t_list)): # different cav
        if i!=0 :
            for j in range(len(t_list[i])): # different time
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
                pairwise_t_matrix[i, j] = t_matrix
    pairwise_t_matrix = torch.tensor(pairwise_t_matrix).to(past_k_lidar_pose.device)
    return pairwise_t_matrix


class PointPillarWhere2commFlowTwoStageWDir(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commFlowTwoStageWDir, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = MaxResNetBEVBackbone(args['base_bev_backbone'], 64)
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

        # if args['compression'] > 0:
        #     self.compression = True
        #     self.naive_compressor = NaiveCompressor(256, args['compression'])

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

        self.design_mode = 0
        if 'design_mode' in args.keys():
            self.design_mode = args['design_mode']
            print(f'=== design mode : {self.design_mode} ===')

        self.noise_flag = 0
        if 'noise' in args.keys():
            self.noise_flag = True
            self.noise_level = {'pos_std': args['noise']['pos_std'], 'rot_std': args['noise']['rot_std'], 'pos_mean': args['noise']['pos_mean'], 'rot_mean': args['noise']['rot_mean']}
            print(f'=== noise level : {self.noise_level} ===')

        self.num_roi_thres = -1
        if 'num_roi_thres' in args.keys():
            self.num_roi_thres = args['num_roi_thres']
            print(f'=== num_roi_thres : {self.num_roi_thres} ===')

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
            self.fused_cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.fused_reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1) # BIN_NUM = 2， # 384
            self.fused_dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.use_dir = False

        if self.design_mode == 0:
            self.matcher = Matcher('flow')
        else:
            self.matcher = Matcher('linear')

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

        self.viz_bbx_flag = False
        if 'viz_bbx_flag' in args.keys() and args['viz_bbx_flag']:
            self.viz_bbx_flag = True
        
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

        for p in self.matcher.parameters():
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

    def bandwidth_filter(self, input, num_box):
        topk_idx = torch.argsort(input['scores'], descending=True)[:num_box]

        output = {}
        output['scores'] = input['scores'][topk_idx]
        output['pred_box_3dcorner_tensor']  = input['pred_box_3dcorner_tensor'][topk_idx]
        output['pred_box_center_tensor'] = input['pred_box_center_tensor'][topk_idx]

        return output

    def generate_box_flow(self, data_dict, pred_dict, dataset, shape_list, device): 
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
        dm_single_list = pred_dict['dm_single_list']

        # H, W = psm_single_list[0].shape[-2:]
        # shape_list = torch.tensor([64, H, W]).to(device)
        
        trans_mat_pastk_2_past0_batch = []
        B = len(lidar_pose_batch)
        box_flow_map_list = []
        reserved_mask_list = []

        if self.viz_bbx_flag:
            ori_reserved_mask_list = []
            single_box_results = None
        
        # for all batches
        for b in range(B):
            box_results = OrderedDict()
            psm_single = psm_single_list[b].reshape(-1, self.k, 2, psm_single_list[b].shape[-2], psm_single_list[b].shape[-1]) # (N_b, k, 2, H, W)
            rm_single = rm_single_list[b].reshape(-1, self.k, 14, rm_single_list[b].shape[-2], rm_single_list[b].shape[-1]) # (N_b, k, 14, H, W)
            if self.use_dir:
                dm_single = dm_single_list[b].reshape(-1, self.k, 4, dm_single_list[b].shape[-2], dm_single_list[b].shape[-1]) # (N_b, k, 4, H, W)

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
            for cav_idx in range(data_dict['record_len'][b]):
                # generate one cav's trans_mat_pastk_2_past0
                pastk_trans_mat_pastk_2_past0 = []
                for i in range(self.k):
                    unit_mat = x1_to_x2(lidar_pose_batch[b][cav_idx, i, :].cpu().numpy(), lidar_pose_batch[b][cav_idx, 0].cpu()) # (4, 4)
                    pastk_trans_mat_pastk_2_past0.append(unit_mat)
                pastk_trans_mat_pastk_2_past0 = torch.from_numpy(np.stack(pastk_trans_mat_pastk_2_past0, axis=0)).to(device) # (k, 4, 4)
                
                m_single = {}
                m_single['psm_single'] = psm_single[cav_idx]
                m_single['rm_single'] = rm_single[cav_idx]
                if self.use_dir:
                    m_single['dm_single'] = dm_single[cav_idx]
                # 1. generate one cav's box results
                box_results[cav_idx] = dataset.generate_pred_bbx_frames(m_single, pastk_trans_mat_pastk_2_past0, cav_past_k_time_diff[cav_idx*self.k:cav_idx*self.k+self.k], anchor_box)

            cav_trans_mat_pastk_2_past0.append(pastk_trans_mat_pastk_2_past0)
            
            # 2. generate box flow in one batch
            if self.viz_bbx_flag:
                box_flow_map, mask, ori_mask, matched_idx_list, compensated_results_list = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag)
                ori_reserved_mask_list.append(ori_mask)
            else:
                box_flow_map, mask = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag)
            box_flow_map_list.append(box_flow_map)
            reserved_mask_list.append(mask)

            if self.viz_bbx_flag:
                single_box_results = box_results
        
        final_flow_map = torch.concat(box_flow_map_list, dim=0)
        final_reserved_mask = torch.concat(reserved_mask_list, dim=0)

        if self.viz_bbx_flag:
            ori_reserved_mask = torch.concat(ori_reserved_mask_list, dim=0)
            return final_flow_map, final_reserved_mask, ori_reserved_mask, single_box_results, matched_idx_list, compensated_results_list

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
        
        shape_list = torch.tensor(batch_dict['spatial_features'].shape[-3:]).to(pairwise_t_matrix.device)

        noise_pairwise_t_matrix = None
        if self.noise_flag and B==1:
            # noise_level = {'pos_std': 0.5, 'rot_std': 0, 'pos_mean': 0, 'rot_mean': 0}
            noise_pairwise_t_matrix = get_past_k_pairwise_transformation2ego(data_dict['past_lidar_pose'], self.noise_level, k=self.k, max_cav=5)[:record_len[0]].unsqueeze(0)

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
        psm_single = self.cls_head(spatial_features_2d).detach()
        rm_single = self.reg_head(spatial_features_2d).detach()

        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d)

        single_detection_bbx = None

        # if self.design_mode != 4 and not self.only_tune_header_flag:
        if self.design_mode != 4:
            # generate box flow
            # [B, 256, 50, 176]
            single_output = {}
            single_output.update({'psm_single_list': self.regroup(psm_single, record_len, k), 
            'rm_single_list': self.regroup(rm_single, record_len, k)})
            if self.use_dir:
                single_output.update({
                    'dm_single_list': self.regroup(dm_single, record_len, k)
                })
            if self.viz_bbx_flag:
                box_flow_map, reserved_mask, ori_reserved_mask, single_detection_bbx, matched_idx_list, compensated_results_list = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device)
            else:
                box_flow_map, reserved_mask = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device)

        if 'flow_gt' in data_dict['label_dict']:
            flow_gt = data_dict['label_dict']['flow_gt']
            mask_gt = data_dict['label_dict']['warp_mask']
        else:
            flow_gt = None
        
        # if self.only_tune_header_flag:
        #     box_flow_map = flow_gt
        #     reserved_mask = mask_gt

        # rain attention:
        if self.multi_scale:
            if self.design_mode == 0 or self.design_mode==5:
                if self.viz_bbx_flag:
                    fused_feature, communication_rates, result_dict, single_updated_feature = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map, reserved_mask=reserved_mask,
                    flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag)
                else:
                    fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'],
                        psm_single,
                        record_len,
                        pairwise_t_matrix, 
                        record_frames,
                        self.backbone,
                        [self.shrink_conv, self.cls_head, self.reg_head],
                        box_flow=box_flow_map, reserved_mask=reserved_mask,
                        flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag, noise_pairwise_t_matrix=noise_pairwise_t_matrix)
            elif self.design_mode == 4:
                fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head])
            else: 
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
        if self.only_tune_header_flag:
            psm = self.fused_cls_head(fused_feature)
            rm = self.fused_reg_head(fused_feature)
        else: 
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)
            # psm = self.fused_cls_head(fused_feature)
            # rm = self.fused_reg_head(fused_feature)
        
        # fuse 之后的 feature (ego)
        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_dir:
            # dm = self.dir_head(fused_feature)
            dm = self.fused_dir_head(fused_feature)
            output_dict.update({'dm': dm})

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

        if self.viz_bbx_flag:
            output_dict.update({
                'single_detection_bbx': single_detection_bbx, 
                'matched_idx_list': matched_idx_list, 
                'compensated_results_list': compensated_results_list
            })
            _, C, H, W = batch_dict['spatial_features'].shape
            output_dict.update({
                'single_updated_feature': single_updated_feature,
                'single_original_feature': batch_dict['spatial_features'].reshape(-1, self.k, C, H, W)[:, 0, :, :, :], 
                'single_flow_map': box_flow_map, 
                'single_reserved_mask': reserved_mask, 
                'single_original_reserved_mask': ori_reserved_mask
            })

        if self.design_mode == 1:
            output_dict.update({'flow_recon_loss': flow_recon_loss})
        
        output_dict.update(result_dict) 
        
        return output_dict
