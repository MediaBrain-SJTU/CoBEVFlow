# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import easydict
import copy
import math
from torch.autograd import Variable
import torch.nn.functional as F
import random
# from dgl.nn.pytorch.factory import KNNGraph
# import dgl
import numpy as np
import os

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
from opencood.models.sub_modules.convgru import ConvGRU
from icecream import ic
from matplotlib import pyplot as plt
# from opencood.models.sub_modules.SyncLSTM import SyncLSTM
from opencood.models.sub_modules.MotionNet import STPN, MotionPrediction, StateEstimation, FlowUncPrediction
# from opencood.models.sub_modules.dcn_net import DCNNet
from opencood.models.comm_modules.where2comm_multisweep import Communication

import torch.nn as nn

class FineTuneFlow(nn.Module):
    def __init__(self):
        super(FineTuneFlow, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        return x

class FineTuneFlow_2(nn.Module):
    def __init__(self):
        super(FineTuneFlow_2, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_warped_feature_mask(flow, updated_grid):
    ''' 
    get the mask of the warped feature map, where the value is 1 means the location is valid, and 0 means the location is invalid.
    ----------
    Args:
        flow: (2, H, W)
        updated_grid: (2, H, W)

    Returns:
        mask: (H, W)
    '''
    def get_large_flow_indices(flow, threshold=0):
        '''
        find the indices of the flow map where the flow is larger than the threshold, which means these locations have been moved.
        ----------
        Args:
            flow: (2, H, W)
            
        '''
        max_values, max_indices = torch.max(torch.abs(flow[:2]), dim=0)
        large_indices = torch.nonzero(max_values > threshold, as_tuple=False)
        return large_indices

    def remove_duplicate_points(points):
        unique_points, inverse_indices = torch.unique(points, sorted=True, return_inverse=True, dim=0)
        return unique_points, inverse_indices
    
    def get_nonzero_idx(flow, idx):
        # print(idx[:, 0].max())
        # print(idx[:, 1].max())
        flow_values = flow[:, idx[:, 0], idx[:, 1]]
        nonzero_idx = torch.nonzero(torch.abs(flow_values).sum(dim=0) == 0, as_tuple=False).squeeze()
        return idx[nonzero_idx]
    
    def filter_tensor(A, H, W):
        mask = (A[:, 0] < H) & (A[:, 1] < W)
        filtered_A = A[mask]
        return filtered_A
    
    flow_idx = get_large_flow_indices(flow)

    _, H, W = flow.shape

    mask = torch.ones(H, W)
    if flow_idx.shape[0] == 0:
        return mask

    # print(flow_idx)
    updated_grid_points_tmp = updated_grid[:, flow_idx[:,0], flow_idx[:,1]].to(torch.int64).T
    # change the order of dim 1
    updated_grid_points = torch.zeros_like(updated_grid_points_tmp)
    updated_grid_points[:, 0] = updated_grid_points_tmp[:, 1]
    updated_grid_points[:, 1] = updated_grid_points_tmp[:, 0]
    # print(updated_grid_points)

    unique_points_idx, _ = remove_duplicate_points(updated_grid_points)
    # print(unique_points_idx)
    
    # mast out the out of range indices
    unique_points_idx = filter_tensor(unique_points_idx, H, W)
    
    nonzero_idx = get_nonzero_idx(flow, unique_points_idx)
    # print(nonzero_idx)

    if len(nonzero_idx.shape) > 1:
        mask[nonzero_idx[:, 0], nonzero_idx[:, 1]] = 0
    else: 
        mask[nonzero_idx[0], nonzero_idx[1]] = 0

    return mask

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]
        
class raindrop_fuse(nn.Module):
    def __init__(self, args, design_mode=0):
        super(raindrop_fuse, self).__init__()
        
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']

        #################################################
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                else:
                    # self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else: 
            if self.agg_mode == 'MAX': # max fusion, debug use
                self.fuse_modules = MaxFusion()

        self.design = design_mode
        if self.design == 1:
            self.fine_conv = FineTuneFlow()
        elif self.design == 2:
            self.stpn = STPN(args['channel_size'])
            self.motion_pred = MotionPrediction(seq_len=1)
            self.state_classify = StateEstimation(motion_category_num=1)
            self.fine_conv = FineTuneFlow()
            self.fine_conv_2 = FineTuneFlow_2()
        else:
            self.stpn = STPN(args['channel_size'])
            self.motion_pred = MotionPrediction(seq_len=1)
            self.state_classify = StateEstimation(motion_category_num=1)

    def regroup(self, x, len, k=0):
        """
        split into different batch and time k

        Parameters
        ----------
        x : torch.Tensor
            input data, (B, ...)

        len : list 
            cav num in differnt batch, eg [3, 3]

        k : torch.Tensor
            num of past frames
        
        Returns:
        --------
        split_x : list
            different cav's lidar feature
            for example: k=4, then return [(3x4, C, H, W), (2x4, C, H, W), ...]
        """
        cum_sum_len = torch.cumsum(len*k, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def generateFlow(self, feats, pairwise_t_matrix, record_len, flow_gt=None):
        '''
        1. generate flow from feature sequence
        2. then update the feature
        Note:
            1. ego feature does not need to be updated

        params:
            feats: (sum(B,N,K), C, H, W)
            pairwise_t_matrix: (B, L, K, 2, 3)
            record_len: (B)

        return:
            flow: 
        '''
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, 2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
            node_features = batch_node_features[b]
            # neighbor_feature = warp_affine_simple(node_features, t_matrix, (H, W))
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            
            # # MotionNet generating flow
            # bevs = self.stpn(node_features)
            # flow = self.motion_pred(bevs) # [N, 2, H, W]
            # flow_list.append(flow)
            # # Motion State Classification head
            # state_class_pred = self.state_classify(bevs)
            # state_class_pred_list.append(state_class_pred)

            # TODO: use gt flow for debug use
            flow = batch_flow_gt[b].view(-1, 2, H, W)  # (N, 2, H, W)
            flow_list.append(flow)
            state_class_pred = torch.ones(1, H, W)
            state_class_pred_list.append(state_class_pred)
            ##########################################

            # # Original flow warp code
            # # Given disp shift feature
            # x_coord = torch.arange(W).float()   # [0, ..., W]
            # y_coord = torch.arange(H).float()   # [0, ..., H]
            # y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            # grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1, -1).to(flow.device)  # N, 2, H, W
            # # updated_grid = grid + flow * (state_class_pred.sigmoid() > self.flow_thre)
            # updated_grid = grid + flow
            
            # # TODO: debug use, do not use gt flow, keep the original delay feature as the fusion input. 
            # # updated_grid = grid
            
            # # generate the mask for filtering out the pixels which moved to other locations but not filled by others
            # mask_list = []
            # for i in range(flow.shape[0]):
            #     mask_list.append(get_warped_feature_mask(flow[i], updated_grid[i]))
            # mask = torch.stack(mask_list, dim=0)
            # mask = mask.unsqueeze(1).repeat(1, C, 1, 1).to(flow.device)
            #################################
            
            # updated_grid[:, 0, :, :] = updated_grid[:, 0, :, :] / (W / 2.0) - 1.0
            # updated_grid[:, 1, :, :] = updated_grid[:, 1, :, :] / (H / 2.0) - 1.0
            # latest_node_features = node_features[:, 0, :, :, :] # (N, C, H, W)
            # updated_features = F.grid_sample(latest_node_features, grid=updated_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
            
            # use szwei flow generator
            latest_node_features = node_features[:, 0, :, :, :] # (N, C, H, W)
            updated_features = F.grid_sample(latest_node_features, grid=flow.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
            
            # mask the features
            # updated_features = mask * updated_features
            # ego feature use the latest feature (no delay)
            updated_features[0, :, :, :] = latest_node_features[0, :, :, :]

            updated_features_list.append(updated_features)
        
        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        # TODO: for debug use
        # debug_path = '/remote-home/share/sizhewei/logs/where2comm_flow_debug/viz_flow/'
        # torch.save(feats, debug_path + 'feats.pt')
        # torch.save(updated_features_all, debug_path + 'updated_features_all.pt')

        flow_all = torch.cat(flow_list, dim=0)  # (sum(B,N), 2, H, W)
        state_class_pred_all = torch.cat(state_class_pred_list, dim=0)  # (sum(B,N), 2, H, W)
        
        return updated_features_all, flow_all, state_class_pred_all

    def update_features_boxflow(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        batch_flow_map = self.regroup(flow_map, record_len, k=1)
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, k=2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            flow = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            flow = self.fine_conv(flow) # [N, H, W, 2]
            flow_list.append(flow)
            # # motion net flow
            # bevs = self.stpn(node_features)
            # flow_pred = self.motion_pred(bevs) # [N, 2, H, W]
            # # Motion State Classification head
            # state_class_pred = self.state_classify(bevs)
            # state_class_pred_list.append(state_class_pred)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 
            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='nearest', align_corners=False)

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            updated_features_list.append(updated_features)

            # normalizing GT flow: gt_flow_map [N, H, W, 2]
            # Given disp shift feature
            x_coord = torch.arange(W).float()   # [0, ..., W]
            y_coord = torch.arange(H).float()   # [0, ..., H]
            y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(N, -1, -1, -1).to(flow.device)
            gt_flow_delta = batch_flow_gt[b].view(-1, 2, H, W)
            gt_flow_map = grid - gt_flow_delta
            gt_flow_map[:, 0, :, :] = gt_flow_map[:, 0, :, :] / (W / 2.0) - 1.0
            gt_flow_map[:, 1, :, :] = gt_flow_map[:, 1, :, :] / (H / 2.0) - 1.0
            gt_flow_map = gt_flow_map.permute(0, 2, 3, 1) 
            gt_flow_map_list.append(gt_flow_map)

        flow_pred = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # compute the flow map loss:
        loss = F.smooth_l1_loss(flow_pred, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        return updated_features_all, loss
    
    def update_features_boxflow_design_2(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        batch_flow_map = self.regroup(flow_map, record_len, k=1)
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, k=2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            flow_box = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            # flow_box = self.fine_conv(flow_box) # [N, H, W, 2]
            
            # motion net flow
            bevs = self.stpn(node_features)
            flow_delta_pred = self.motion_pred(bevs) # [N, 2, H, W]
            # Motion State Classification head
            state_class_pred = self.state_classify(bevs)
            state_class_pred_list.append(state_class_pred)
            # Given disp shift feature
            x_coord = torch.arange(W).float()   # [0, ..., W]
            y_coord = torch.arange(H).float()   # [0, ..., H]
            y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(N, -1, -1, -1).to(flow_box.device)
            # flow_pred = grid - flow_delta_pred
            # # normalizing flow: flow_pred [N, H, W, 2]
            # flow_pred[:, 0, :, :] = flow_pred[:, 0, :, :] / (W / 2.0) - 1.0
            # flow_pred[:, 1, :, :] = flow_pred[:, 1, :, :] / (H / 2.0) - 1.0
            # flow_pred = flow_pred.permute(0, 2, 3, 1) # [N, H, W, 2]

            # normalizing flow_delta_pred 
            flow_delta_pred[:, 0, :, :] = flow_delta_pred[:, 0, :, :] / (W / 2.0) - 1.0
            flow_delta_pred[:, 1, :, :] = flow_delta_pred[:, 1, :, :] / (H / 2.0) - 1.0
            flow_delta_pred = flow_delta_pred.permute(0, 2, 3, 1) # [N, H, W, 2]
            flow = self.fine_conv(flow_delta_pred + flow_box) # [N, H, W, 2]

            # flow = self.fine_conv_2(flow_box, flow_pred) # [N, H, W, 2]
            flow_list.append(flow)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 
            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='nearest', align_corners=False)

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            updated_features_list.append(updated_features)

            # normalizing GT flow: gt_flow_map [N, H, W, 2]
            gt_flow_delta = batch_flow_gt[b].view(-1, 2, H, W)
            gt_flow_map = grid - gt_flow_delta
            gt_flow_map[:, 0, :, :] = gt_flow_map[:, 0, :, :] / (W / 2.0) - 1.0
            gt_flow_map[:, 1, :, :] = gt_flow_map[:, 1, :, :] / (H / 2.0) - 1.0
            gt_flow_map = gt_flow_map.permute(0, 2, 3, 1) 
            gt_flow_map_list.append(gt_flow_map)

        flow_all = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        state_class_pred_all = torch.cat(state_class_pred_list, dim=0)  # (sum(B,N), 2, H, W)
        gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # compute the flow map loss:
        loss = F.smooth_l1_loss(flow_all, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        return updated_features_all, loss, flow_all, state_class_pred_all

    def forward(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None, flow_gt=None, box_flow=None, reserved_mask=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
        
        rm : torch.Tensor
            confidence map, (sum(n_cav), 2, H, W)

        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
        
        flow_gt: ground truth flow, generate by object center and id

        box_flow: flow generate by past 2 frames detection boxes
            
        Returns
        -------
        Fused feature
        flow_map loss
        """
        _, C, H, W = x.shape
        B, L, K = pairwise_t_matrix.shape[:3]

        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # 2. feature compensation with flow
        # 2.1 generate flow, 在同一个坐标系内，计算每个cav的flow
        # 2.2 compensation
        # x: (BxNxK, C, H, W) -> (BxN, C, H, W)
        if self.design == 1:
            updated_features, flow_recon_loss = self.update_features_boxflow(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 2:
            updated_features, flow_recon_loss, flow, state_preds = self.update_features_boxflow_design_2(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        else:
            updated_features, flow, state_preds = self.generateFlow(x, pairwise_t_matrix, record_len, flow_gt) # (BxN, C, H, W)
        
        # 3. feature fusion
        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(updated_features)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)  # (BxN, C', H, W)

                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
                        batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...]
                        _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
                        # x = x * communication_masks_tensor
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2)
                        # TODO: scale = 1, 2 不加 mask
                        # x = x * communication_masks_tensor
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                batch_node_features = self.regroup(x, record_len, 1)

                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix,
                                                    (H, W))
                    record_frames = np.ones((N))*K
                    # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                    
                    if self.agg_mode == 'RAIN':
                        # for sensor embedding
                        sensor_dist = -1# (B, H, W)
                        x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])
                    else: # ATTEN, MAX, Transformer
                        x_fuse.append(self.fuse_modules[i](neighbor_feature))
                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])

                x_fuse = torch.stack(x_fuse)

                ############ 4. Deconv ####################################
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_confidence_maps = self.regroup(rm, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                # _, (B, 1, H, W), float
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix,
                                                (H, W))
                record_frames = np.ones((N))*K
                # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                if self.agg_mode == "RAIN":
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        
        if self.design == 1:
            return x_fuse, communication_rates, {}, flow_recon_loss
        elif self.design == 2:
            return x_fuse, communication_rates, {}, flow_recon_loss #, flow, state_preds
        else:
            return x_fuse, communication_rates, {}, flow, state_preds

class TemporalFusion(nn.Module):
    def __init__(self, args):
        super(TemporalFusion, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        # channel_size = args['channel_size']
        # spatial_size = args['spatial_size']
        # TM_Flag = args['TM_Flag']
        # compressed_size = args['compressed_size']
        # self.compensation_net = SyncLSTM(channel_size, spatial_size, TM_Flag, compressed_size)
        self.stpn = STPN(height_feat_size=args['channel_size'])
        self.flow_thre = args['flow_thre']
        self.motion_pred = MotionPrediction(seq_len=1)
        self.state_classify = StateEstimation(motion_category_num=1)

        self.flow_unc_flag = False
        if 'flow_unc_flag' in args:
            self.flow_unc_flag = True
            self.flow_unc_pred = FlowUncPrediction(seq_len=1)

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        if 'FlowPredictionFix' in args.keys() and args['FlowPredictionFix']:
            self.FlowPredictionFix()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def FlowPredictionFix(self):
        print('------ Flow Prediction Fix, Only train DCN ------')
        for p in self.stpn.parameters():
            p.requires_grad = False
        for p in self.motion_pred.parameters():
            p.requires_grad = False
        for p in self.state_classify.parameters():
            p.requires_grad = False

    def forward(self, x, record_len, pairwise_t_matrix, data_dict=None):
        """
        Fusion forwarding.

        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        feat_seqs = self.regroup(x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        # iteratively warp feature to current timestamp
        batch_feat_seqs = []
        for b in range(B):
            # number of valid timestamps
            K = record_len[b]
            t_matrix = pairwise_t_matrix[b][:K, :K, :, :]
            curr_feat_seq = warp_affine_simple(feat_seqs[b],
                                               t_matrix[0, :, :, :],
                                               (H, W))
            batch_feat_seqs.append(curr_feat_seq[None, ...])
        batch_feat_seqs = torch.cat(batch_feat_seqs, dim=0)  # b, K, c, h, w
        batch_hist_feat_seqs = batch_feat_seqs[:, 1:].flip(1)

        # Backbone network
        bevs = self.stpn(batch_hist_feat_seqs)  # b, K, c, h, w

        # Motion Displacement prediction
        flow = self.motion_pred(bevs)
        flow = flow.view(-1, 2, bevs.size(-2), bevs.size(-1))

        flow_unc = None
        if self.flow_unc_flag:
            flow_unc = self.flow_unc_pred(bevs)
            flow_unc = flow_unc.view(-1, 2, bevs.size(-2), bevs.size(-1))

        # flow = data_dict['flow_gt']

        # Motion State Classification head
        state_class_pred = self.state_classify(bevs)

        # Given disp shift feature
        x_coord = torch.arange(bevs.size(-1)).float()
        y_coord = torch.arange(bevs.size(-2)).float()
        y, x = torch.meshgrid(y_coord, x_coord)
        grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1, -1).to(
            flow.device)
        # updated_grid = grid + flow * (state_class_pred.sigmoid() > self.flow_thre)
        updated_grid = grid - flow
        updated_grid[:, 0, :, :] = updated_grid[:, 0, :, :] / (bevs.size(-1) / 2.0) - 1.0
        updated_grid[:, 1, :, :] = updated_grid[:, 1, :, :] / (bevs.size(-2) / 2.0) - 1.0
        out = F.grid_sample(batch_feat_seqs[:, 1], grid=updated_grid.permute(0, 2, 3, 1), mode='bilinear')
        # out = F.grid_sample(batch_feat_seqs[:,1], grid=updated_grid.permute(0,2,3,1), mode='nearest')
        # out = F.grid_sample(batch_feat_seqs[:,1], grid=grid.permute(0,2,3,1), mode='bilinear')
        # out = batch_feat_seqs[:,1]
        if self.dcn:
            out = self.dcn_net(out)

        flow_dict = {'flow':flow, 'flow_unc': flow_unc}
        return flow_dict, state_class_pred, out

    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding
        Used for debug and visualization


        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        origin_x: torch.Tensor
            pillars (sum(n_cav), C, H * downsample_rate, W * downsample_rate)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        from matplotlib import pyplot as plt

        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L, 1, H, W)).to(x)
                roi_mask[b, i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :], (H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        # visualize warped feature map
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0  # ego
            mask = roi_mask[b, i, :N, ...]
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                  t_matrix[i, :, :, :],
                                                  (H, W))
            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
                plt.clf()
                plt.imshow(mask[idx][0].detach().cpu().numpy())
                plt.savefig(
                    f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/mask_feature_{b}_{idx}")
                plt.clf()

        # visualize origin pillar feature
        origin_node_features = split_origin_x

        for b in range(B):
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            i = 0  # ego
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(origin_node_features[b],
                                                  t_matrix[i, :, :, :],
                                                  (H * self.downsample_rate, W * self.downsample_rate))

            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_{b}_{idx}")
                plt.clf()