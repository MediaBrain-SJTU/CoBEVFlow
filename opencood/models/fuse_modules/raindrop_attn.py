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
from dgl.nn.pytorch.factory import KNNGraph
import dgl
import numpy as np
import os

import torch.nn.functional as F
from turtle import update

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.comm_modules.where2comm_multisweep import Communication

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_single_neighbors(h,w,neighbor_range):
    row = torch.arange(w-neighbor_range,w+neighbor_range+1).repeat(neighbor_range*2+1)
    col = (torch.arange(h-neighbor_range, h+neighbor_range+1)*w).repeat_interleave(neighbor_range*2+1)

    return row+col

class PositionalEncoding_spatial(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model_out, dropout, max_len=300,mode='cat'):
        super(PositionalEncoding_spatial, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = int(d_model_out/2)
        # Compute the positional encodings once in log space.
        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe_ = pe.unsqueeze(0)#batch,len,feature
        pe_h = pe_.unsqueeze(2).repeat(1,1,max_len,1)
        pe_w = pe_.unsqueeze(1).repeat(1,max_len,1,1)
        pe = torch.cat((pe_h,pe_w),dim=-1)

        self.register_buffer('pe', pe)

        self.linear = nn.Linear(2*d_model,d_model)
        self.mode = mode

    def forward(self, x):

        if self.mode == 'cat':
            x = self.linear(torch.cat((x,Variable(pe[:,:x.shape[1],:x.shape[2]],require_grad=False)),dim=-1))

        elif self.mode == 'init':
            x = Variable(pe[:,:x.shape[1],:x.shape[2]],requires_grad=False)

        else:
            x = x + Variable(pe[:,:x.shape[1],:x.shape[2]],requires_grad=False)

        return self.dropout(x)


class PositionalEncoding_sensor_dist(nn.Module):

    def __init__(self, d_model, dropout):
        super(PositionalEncoding_sensor_dist, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model,d_model)
        self.mode = 'cat'
         
    def forward(self, x, sensor_dist):#x.shape=[batch,H,W,F],senfor_dist.shape=[batch,H,W]

        x = x.reshape(-1,x.shape[-1])
        sensor_dist = sensor_dist.reshape(-1)

        pe = torch.zeros(x.shape[0], self.d_model).to(x.device)
        position = sensor_dist.unsqueeze(1) 
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).to(x.device)
         
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)   

        if self.mode == 'cat':
            x = self.linear(torch.cat((x,Variable(pe,require_grad=False)),dim=-1))

        else:
            x = x + Variable(pe,requires_grad=False)
        
        return self.dropout(x)

class PositionalEncoding_irregular_time(nn.Module):

    def __init__(self, d_model, dropout):
        super(PositionalEncoding_irregular_time, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model,d_model)
        self.mode = 'cat'

    def forward(self, x, time):#x.shape=[batch,height,width,feature],time.shape=[batch]

        pe = torch.zeros(x.shape[0],self.d_model).to(x.device)
        position = time.unsqueeze(1) 
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).to(x.device)
         
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)   

        pe = pe.unsqueeze(1).unsqueeze(1)

        if self.mode == 'cat':
            x = self.linear(torch.cat((x,Variable(pe,require_grad=False)),dim=-1))

        else:
            x = x + Variable(pe,requires_grad=False)

        
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    

    mask = mask.transpose(1,2).to(query.device)
    # print('score',scores.shape,'mask',mask.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10) 
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, q_dim, k_dim,v_dim, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  
        self.h = h 
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear_q = nn.Linear(q_dim,d_model)
        self.linear_k = nn.Linear(k_dim,d_model)
        self.linear_v = nn.Linear(v_dim,d_model)
        self.linear_out = nn.Linear(d_model,q_dim)
        self.attn = None  
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 

        query = self.linear_q(query).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
        key = self.linear_k(key).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
        value = self.linear_v(value).view(nbatches,-1,self.h,self.d_k).transpose(1,2)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches ,-1, self.h * self.d_k)


        return self.linear_out(x)

class MultiHeadedAttention_spatial(nn.Module):
    def __init__(self, h, q_dim, k_dim,v_dim, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_spatial, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  
        self.h = h 
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear_q = nn.Linear(q_dim,d_model)
        self.linear_k = nn.Linear(k_dim,d_model)
        self.linear_v = nn.Linear(v_dim,d_model)
        self.linear_out = nn.Linear(d_model,q_dim)
        self.attn = None  
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, spatial_neighbors , mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query = self.linear_q(query).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
        key_ = self.linear_k(query).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
        value_ = self.linear_v(query).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
        
        key = torch.index_select(key_,dim=0,index=spatial_neighbors.reshape(-1))
        key = key.squeeze(1).reshape(nbatches,spatial_neighbors.shape[1],self.h,self.d_k)

        value = torch.index_select(value_,dim=0,index=spatial_neighbors.reshape(-1))
        value = value.squeeze(1).reshape(nbatches,spatial_neighbors.shape[1],self.h,self.d_k)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches ,-1, self.h * self.d_k)

        return self.linear_out(x)


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, neighbor_range, height, width, dropout, max_len=300):
        super(AttentionFusion, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.neighbor_range = neighbor_range
        self.height = height
        self.width = width

        self.spatial_query = PositionalEncoding_spatial(hidden_dim, dropout, max_len, mode='init')

        self.spatial_encoding = PositionalEncoding_spatial(hidden_dim, dropout, max_len)

        self.time_encoding = PositionalEncoding_irregular_time(hidden_dim, dropout)

        self.sensor_dist_encoding = PositionalEncoding_sensor_dist(hidden_dim, dropout)

        self.layers_temporal = clones(MultiHeadedAttention(self.num_heads, self.input_dim, self.input_dim, self.input_dim, self.hidden_dim, dropout),num_layers)
        self.temporal_aggregator = MultiHeadedAttention(self.num_heads, self.input_dim, self.input_dim, self.input_dim, self.hidden_dim, dropout)
        self.layers_spatial = clones(MultiHeadedAttention_spatial(self.num_heads, self.input_dim, self.input_dim, self.input_dim, self.hidden_dim, dropout),num_layers)
        
        self.generate_spatial_neighbors(self.height, self.width, self.neighbor_range)
        print("================Rain attention Init================")

    def generate_spatial_neighbors(self, height, width, neighbor_range):
        self.spatial_neighbors = torch.zeros(height*width,(neighbor_range*2+1)**2)

        for h in range(height):
            for w in range(width):
                if h < neighbor_range :
                    if w < neighbor_range:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(neighbor_range,neighbor_range,neighbor_range)
                    elif w > width-neighbor_range-1:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(neighbor_range,width-neighbor_range-1,neighbor_range)
                    else:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(neighbor_range,w,neighbor_range)

                elif h > height-neighbor_range-1:
                    if w < neighbor_range:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(height-neighbor_range-1,neighbor_range,neighbor_range)
                    elif w > width-neighbor_range-1:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(height-neighbor_range-1,width-neighbor_range-1,neighbor_range)
                    else:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(height-neighbor_range-1,w,neighbor_range)

                else:
                    if w < neighbor_range:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(h,neighbor_range,neighbor_range)
                    elif w > width-neighbor_range-1:
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(h,width-neighbor_range-1,neighbor_range)
                    else: 
                        self.spatial_neighbors[h*height+w] = get_single_neighbors(h,w,neighbor_range)


    def forward(self, feartures, sensor_dist, time_stamps, valid_mask):
        """
        Parameters:
        -----------
        features: torch.Tensor, shape (B, C, H, W)
            neighbor features in ego view
            B = \sum_{i}^{num_cav} k_i, num_cav 表示cav数量, k_i 表示对于cav_i 有多少观测帧
            C : feature 维度，一般为 256
            H : height, 一般为 100
            W : width, 一般为 252

        sensor_dist: torch.Tensor, shape (B, H, W)
            distance between each block and the sensor
            
        time_stamps: torch.Tensor, shape (B)
            表示每个位置的时间信息

        valid_mask: (B,H,W)
            为0/1,表示是否在该位置有效

        Returns:
        --------
        output_feature: torch.Tensor, shape (C, H, W)
            ego cav's fused feature at current frame.
        """
        batch_all,feature_dim,height,width = feartures.shape
        features = features.transpose(0,2,3,1)
        
        final_feature = self.spatial_query(None)# final_query.shape = [H,W,F]

        features = self.spatial_encoding(features)# features.shape = [batch_all,height,width,feature_dim]
        if sensor_dist!=-1:
            features = self.sensor_dist_encoding(features,sensor_dist)
        features = self.time_encoding(features,time_stamps)

        valid_mask = valid_mask.reshape(batch_all,height*width)# valid_mask.shape = [batch_all,height*width]
        valid_mask_sum = valid_mask.sum(0)#valid_mask_sum.shape = [h*w]
        max_len = valid_mask_sum[0]

        valid_idx = torch.arange(0,batch_all).unsqueeze(-1).repeat(1,height*width)# valid_idx.shape = [batch_all,height*width]

        valid_idx = torch.where(valid_mask==1,valid_idx,1e10)# valid_idx.shape = [batch_all,height*width]

        valid_idx = torch.sort(valid_idx,dim=0)[:max_len].unsqueeze(-1).repeat(1,1,feature_dim)#valid_idx.shape = [max_len,height*width,feature_dim]

        valid_idx = torch.where(valid_idx==1e10,0,valid_idx)

        temporal_neighbors_set = feartures.reshape(batch_all,height*width,feature_dim).gather(valid_idx,dim=1)#temporal_neighbors_set.shape = [max_len,height*width,feature_dim]

        temporal_attention_mask = torch.zeros(height*width,1,max_len)
        for i in range(height*width):
            temporal_attention_mask[i,0,:valid_mask_sum[i]] = 1

        temporal_neighbors_set = temporal_neighbors_set.transpose(1,0,2)

        for l in self.num_layers:
            temporal_neighbors_set = self.layers_temporal[l](temporal_neighbors_set,temporal_neighbors_set,temporal_neighbors_set,temporal_attention_mask)#shape[h*w,1,feature]

        final_feature = self.temporal_aggregator(final_feature,temporal_neighbors_set,temporal_neighbors_set,temporal_attention_mask)#shape[h*w,1,feature]

        for l in self.num_layers:
            final_feature = self.layers_spatial[l](final_feature,self.spatial_neighbors)#shape[h*w,1,feature]

            

        return final_feature

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

        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            # self.fuse_modules = nn.ModuleList()

        if self.agg_mode == 'MAX': # max fusion, debug use
            self.fuse_modules = MaxFusion()
        else: # rain attention
            N = args['N']
            d_model = args['d_model']
            d_ff = args['d_ff']
            h = args['h']
            dropout = args['dropout']
            input_dim = args['input_dim']
            num_layers = args['num_layers']
            neighbor_range = args['neighbor_range']
            max_len = args['max_len']

            # def __init__(self, input_dim, hidden_dim, num_heads, num_layers, neighbor_range, height, width, dropout, max_len=300):
            self.fuse_modules = AttentionFusion(input_dim, d_ff, h, num_layers, neighbor_range, 100, 252, dropout, max_len)
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in self.fuse_modules.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

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
            time interval
            
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

        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                ############ 1. Communication (Mask the features) #########
                if i==0: # TODO: mask conv
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len) # cls_head 出来的 map 
                        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                
                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix[0, :, :, :],
                                                    (H, W))
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
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
                if self.agg_mode == "MAX":
                    x_fuse.append(self.fuse_modules(neighbor_feature))
                else:
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        
        return x_fuse, communication_rates, {}