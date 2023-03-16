# -*- coding: utf-8 -*-
# Author: Yuxi Wei, Sizhe Wei <sizhewei@sjtu.edu.cn> 
# Date: 2022/12/06
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

import torch.nn.functional as F
from turtle import update
# import ipdb
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.comm_modules.where2comm_multisweep import Communication

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]
        
class raindrop_fuse(nn.Module):
    def __init__(self, args):
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
        if self.agg_mode == 'MAX': # max fusion, debug use
            self.fuse_modules = MaxFusion()

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

    def forward(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None):
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
            shape: (B, L, L, 4, 4) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
            
        Returns
        -------
        Fused feature.
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
        
        return x_fuse, communication_rates, {}

