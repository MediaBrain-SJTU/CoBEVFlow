# -*- coding: utf-8 -*-
# Author: Yuxi Wei, Sizhe Wei <sizhewei@sjtu.edu.cn> 
# Date: 2022/12/06
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
where2comm + swin transformer + single supervisation.
"""

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

from opencood.models.fuse_modules.STPT import STPT

if_save_pt = False  # TODO: for debug use, save pt file
pt_save_path = '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out'

class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1


class mask_fuse(nn.Module):
    def __init__(self, channel):
        super(mask_fuse, self).__init__()
        self.weight_conv = PixelWeightedFusionSoftmax(channel)
        self.soft_max = nn.Softmax(dim=0)
    def forward(self, ego_feature, nonego_feature):#####[cars, C, H, W]
        ''' 
        ego_feature [1, C, H, W]
        nonego_feature [n, C, H, W]
        '''
        all_feature = torch.cat((ego_feature,nonego_feature),dim=0)
        ego_feature_repeat = ego_feature.repeat(nonego_feature.shape[0]+1,1,1,1)

        cat_feature = torch.cat((ego_feature_repeat,all_feature),dim=1)

        mask_map = self.weight_conv(cat_feature)
        mask_weight = self.soft_max(mask_map)

        final_feature = torch.sum(mask_weight*all_feature, dim=0)

        return final_feature

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding_spatial(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model_feature, d_model_hidden, dropout, max_len=300,mode='cat'):
        super(PositionalEncoding_spatial, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = int(d_model_hidden/2)
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

        self.linear = nn.Linear(d_model_feature + d_model_hidden, d_model_feature)
        self.mode = 'plus'

    def forward(self, x):
        if self.mode == 'cat':
            tmp = Variable(self.pe[:,:x.shape[1],:x.shape[2]],requires_grad=False)
            tmp = tmp.repeat(x.shape[0], 1, 1, 1)
            x = self.linear(torch.cat((x, tmp),dim=-1))

        elif self.mode == 'init':
            x = Variable(self.pe[:,:x.shape[1],:x.shape[2]],requires_grad=False)

        else:
            tmp = Variable(self.pe[:,:x.shape[1],:x.shape[2]],requires_grad=False)
            tmp = tmp.repeat(x.shape[0], 1, 1, 1)
            x = x + tmp

        return self.dropout(x)

class PositionalEncoding_spatial_custom(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model_feature, d_model_hidden, dropout,mode='cat'):
        super(PositionalEncoding_spatial_custom, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = int(d_model_hidden/2)
        # Compute the positional encodings once in log space.
        
        self.linear = nn.Linear(d_model_feature + d_model_hidden, d_model_feature)
        self.mode = 'plus'

    def forward(self, x, nonego_idx):#x(batch,num,feature)
   
        pe_h = torch.zeros((x.shape[1], self.d_model)).to(x.device)
        pe_w = torch.zeros((x.shape[1], self.d_model)).to(x.device)
        position_h, position_w = nonego_idx[0],nonego_idx[1]
        position_h = position_h.unsqueeze(1)
        position_w = position_w.unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).to(position_h.device)
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)

        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
       
        pe = torch.cat((pe_h,pe_w),dim=-1).unsqueeze(0).repeat(x.shape[0],1,1)

        if self.mode == 'cat':
            x = self.linear(torch.cat((x, pe),dim=-1))

        else:
            x = x + Variable(pe,requires_grad=False)

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

    def __init__(self, d_model_feature, d_model_hidden, dropout):
        super(PositionalEncoding_irregular_time, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model_feature
        self.linear = nn.Linear(d_model_feature+d_model_hidden, d_model_feature)
        self.mode = 'plus'

    def forward(self, x, time):#x.shape=[batch,height,width,feature],time.shape=[batch]

        pe = torch.zeros(x.shape[0],self.d_model).to(x.device)
        position = time.unsqueeze(1) 
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).to(x.device)
         
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)   

        if self.mode == 'cat':
            if len(x.shape) == 4:
                pe = pe.unsqueeze(1).unsqueeze(1)
                tmp = Variable(pe, requires_grad=False)
                tmp = tmp.repeat(1, x.shape[1], x.shape[2], 1)
            else:
                pe = pe.unsqueeze(1)
                tmp = Variable(pe, requires_grad=False)
                tmp = tmp.repeat(1, x.shape[1], 1)
            x = self.linear(torch.cat((x, tmp),dim=-1))

        else:
            if len(x.shape) == 4:
                pe = pe.unsqueeze(1).unsqueeze(1)
                tmp = Variable(pe, requires_grad=False)
                tmp = tmp.repeat(1, x.shape[1], x.shape[2], 1)
            else:
                pe = pe.unsqueeze(1)
                tmp = Variable(pe, requires_grad=False)
                tmp = tmp.repeat(1, x.shape[1], 1)
            x = x + tmp

        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    # ipdb.set_trace()
    
    # print('score',scores.shape,'mask',mask.shape)

    if mask is not None:
        # mask = mask.transpose(1,2).to(query.device)
        mask = mask.to(query.device)
        scores = scores.masked_fill(mask == 0, -1e10) 

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
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

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
 
    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        # print('sub',x.shape,self.dropout(sublayer(self.norm(x))).shape)
        return x + self.dropout(sublayer(self.norm(x))) #残差连接

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model,d_out, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer_temporal(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer_temporal, self).__init__()
        self.self_attn = self_attn #sublayer 1 
        self.feed_forward = feed_forward #sublayer 2 
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #拷贝两个SublayerConnection，一个为了attention，一个是独自作为简单的神经网络
        self.size = size

    def forward(self, q, k, v, mask):
        "Follow Figure 1 (left) for connections."
        # print('x',x.shape,'attn',self.self_attn(x,x,x,mask).shape)
        x = self.sublayer[0](q, lambda q: self.self_attn(q, k, v, mask)) #attention层 对应层1
        return self.sublayer[1](x, self.feed_forward) # 对应 层2

class Spatial_conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Spatial_conv,self).__init__()
        self.conv = nn.Conv2d(in_channels = input_channels,
                              out_channels = output_channels,
                              kernel_size = (kernel_size,kernel_size),
                              stride = (1,1),
                              padding = (int((kernel_size-1)/2),int((kernel_size-1)/2))
        )

        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x):
        init_x = x.clone()
        x = self.conv(x)
        x = self.relu(x)
        # x = self.bn(x)
        return x+init_x

class SpatialAttention_mtf(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_mtf, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, curr, prev):#####(batch, channel, h, w)
        avg_out = torch.mean(curr, dim=1, keepdim=True)
        max_out, _ = torch.max(curr, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * prev ####(batch-1, channel, h, w)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, mask):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask != None:
            score = score.masked_fill(mask == 0, -1e10) 
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttentionFusion(nn.Module):
    def __init__(self, settings, input_dim, hidden_dim, num_heads, num_layers, neighbor_range, height, width, dropout, max_len=300, compensation=True):
        super(AttentionFusion, self).__init__()
        self.sweep_length = settings['sweep_length']
        self.if_max_aggre = settings['if_max_aggre']
        self.if_spatial_encoding = settings['if_spatial_encoding']
        self.if_ego_time_encoding = settings['if_ego_time_encoding']
        self.if_nonego_time_encoding = settings['if_nonego_time_encoding']
        self.if_sensor_encoding = settings['if_sensor_encoding']
        self.if_time_attn_aggre = settings['if_time_attn_aggre']
        self.if_spatial_conv = settings['if_spatial_conv']
        self.if_dotproductattn = settings['if_dotproductattn']
        self.use_mask = settings['use_mask']
        self.confidence_fetch = settings['confidence_fetch']
        self.if_conv_aggre = settings['if_conv_aggre']
        
        self.sup_individual = settings['sup_individual']  # 用于单车feature 进行补偿 算recon loss
        self.if_swin_trans = True

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.neighbor_range = neighbor_range
        self.height = height
        self.width = width
        self.compensation=compensation
        
        self.dilation_size = 0
        if self.compensation:
            if 'dilation_size' in settings:
                self.dilation_size = settings['dilation_size']
            else:
                self.dilation_size = 1
            self.dilate_mask = nn.MaxPool2d(kernel_size=self.dilation_size, stride=1, padding=int((self.dilation_size-1)/2))
            if self.if_spatial_encoding:
                self.spatial_encoding = PositionalEncoding_spatial(self.input_dim, hidden_dim, dropout, max_len)
                self.spatial_encoding_nonego = PositionalEncoding_spatial_custom(self.input_dim, hidden_dim, dropout)
            if self.if_ego_time_encoding or self.if_nonego_time_encoding:
                self.time_encoding = PositionalEncoding_irregular_time(self.input_dim, hidden_dim, dropout)
            if self.if_sensor_encoding:
                self.sensor_dist_encoding = PositionalEncoding_sensor_dist(hidden_dim, dropout)

            c = copy.deepcopy
            temporal_attn = MultiHeadedAttention(self.num_heads, self.input_dim, self.input_dim, self.input_dim, self.hidden_dim, dropout)

            ff = PositionwiseFeedForward(self.input_dim, self.input_dim, self.hidden_dim, dropout)

            if self.if_time_attn_aggre:
                self.layers_temporal = clones(EncoderLayer_temporal(self.input_dim, c(temporal_attn), c(ff), dropout),num_layers)
            # self.temporal_aggregator = MultiHeadedAttention(self.num_heads, hidden_dim, self.input_dim, self.input_dim, self.hidden_dim, dropout)
            if self.if_spatial_conv:  
                self.layers_spatial = clones(Spatial_conv(self.input_dim,self.input_dim,kernel_size=5),num_layers)
            if self.if_swin_trans:
                self.swin_trans = STPT(in_chans=self.input_dim,
                    depths=[2, 2], 
                    num_heads=[2, 2], 
                    window_sizes=[(4,4), (4,4)],
                    num_frames=3,
                    num_out_frames=1,
                    embed_dim=36,  # TODO
                    use_hdmap=False,
                    use_hdmap_feat=False)
            
            # self.layers_spatial = Spatial_conv(self.input_dim,self.input_dim,kernel_size=5)
            if self.if_dotproductattn:
                self.dotproductattn = ScaledDotProductAttention(self.input_dim)
            if self.if_conv_aggre:
                self.conv_aggre = SpatialAttention_mtf()
        
        print(f"===== Dilation size : {self.dilation_size} =====")




    def forward(self, features, sensor_dist, time_stamps, valid_mask):
        """
        Parameters:
        -----------
        features: torch.Tensor, shape (B, C, H, W)
            neighbor features in ego view
            B : 按照顺序排列: ego[curr], cav_1[curr], cav_1[past_1], ..., cav_2[past_k], ....
            C : feature 维度，一般为 256
            H : height, 一般为 100
            W : width, 一般为 252

        sensor_dist: torch.Tensor, shape (B, H, W)
            distance between each block and the sensor
            
        time_stamps: torch.Tensor, shape (B)
            表示每个位置的时间信息

        valid_mask: (B, 1, H, W)
            为0/1,表示是否在该位置有效

        Returns:
        --------
        output_feature: torch.Tensor, shape (C, H, W)
            ego cav's fused feature at current frame.
        """
        if self.if_max_aggre:
            features = torch.max(features, dim=0)[0]
            if self.if_spatial_conv:
                features = features.unsqueeze(0)
                features = self.layers_spatial[0](features)
            return features.squeeze(0)
        if self.confidence_fetch == False:
            if self.sup_individual:
                curr_features = features[1::self.sweep_length+1].clone()    # [n, C, H, W]
                latency_features = features[2::self.sweep_length+1].clone()  # [n, C, H, W]
                if self.compensation==False:
                    all_latency_features = torch.cat((features[0].unsqueeze(0), latency_features), dim=0)
                    return all_latency_features, 0
                    
                    # final_fuse_feature = torch.max(torch.cat((features[:1],curr_features),dim=0),dim=0)[0]
                    # final_fuse_feature_latency = torch.max(torch.cat((features[:1],latency_features),dim=0),dim=0)[0]

                    # return final_fuse_feature_latency,final_fuse_feature,final_fuse_feature_latency,0,0


                cav_num, C, H, W = features.shape
                individual_recon_loss = 0
                latency_recon_loss = 0
                # debug use, set true to save pt for viz
                
                if if_save_pt:
                    torch.save(time_stamps, pt_save_path +'time_stamps'+'.pt')
                    torch.save(features[1:], '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ 'features'+'.pt')
                
                num_nonego = int((cav_num-1)/(self.sweep_length+1)) 
                nonego_features = features[1:].clone()                  # [nonego_cavs(1+k), C, H, W]
                
                nonego_splits = nonego_features.split(self.sweep_length+1,0) # ([1+k, C, H, W], [], ..., [])
                nonego_splits = torch.stack(nonego_splits,dim=0)        # [nonego_cars, 1+k, C, H, W]
                # torch.save(nonego_splits, '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ 'nonego_splits'+'.pt')
                nonego_history = nonego_splits[:,1:]                    # [nonego_cars, k, C, H, W]

                nonego_timestamps = time_stamps[1:]                     # [nonego_cavs(1+k), ]
                nonego_timestamps_splits = nonego_timestamps.split(self.sweep_length+1, 0)
                nonego_timestamps_splits = torch.stack(nonego_timestamps_splits,dim=0)   # [n, 1+k, ]
                nonego_timestamps_history = nonego_timestamps_splits[:, 1:]


                nonego_history = nonego_history.reshape(-1, C, H, W).permute(0,2,3,1)
                nonego_timestamps_history = nonego_timestamps_history.reshape(-1)
                # nonego_features = nonego_features.permute(0,2,3,1)      # 
                nonego_history = self.time_encoding(nonego_history,nonego_timestamps_history)
                nonego_history = nonego_history.permute(0,3,1,2).reshape(num_nonego, self.sweep_length, C, H, W)      # [nonego_cavs, k, C, H, W]

                # mask after TE, valid_mask: [B, H, W]
                non_ego_mask = valid_mask[1:]                           # [nonego_cavs*(k+1), 1, H, W]
                non_ego_mask_split = non_ego_mask.split(self.sweep_length+1,0) # ([1+k, 1, H, W], ... ) nonego_cavs 
                non_ego_mask = torch.stack(non_ego_mask_split, dim=0)   # nonego_cavs, 1+k, H, W
                if if_save_pt:
                    torch.save(non_ego_mask, '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ 'nonego_mask_splits'+'.pt')

                non_ego_mask_past = non_ego_mask[:, 1:]                 # nonego_cavs, k, H, W
                nonego_history = nonego_history * non_ego_mask_past        # [nonego_cavs, k, C, H, W]
                
                # nonego_splits = nonego_splits.permute(0,3,4,1,2).reshape(num_nonego*H*W, self.sweep_length, C)
                # time_attned_feature = self.layers_temporal[0](nonego_splits, nonego_splits, nonego_splits, None)
                # time_attned_feature = nonego_splits
                # time_attned_feature = time_attned_feature.reshape(num_nonego, H, W, self.sweep_length*C).permute(0,3,1,2) 
                
                hidden = self.swin_trans(nonego_history)                # [non_ego_cars, 1, C, H, W]
                # query = self.predict_query.repeat(num_nonego,1,1)
                # final_estimations = self.predict_attention(query, hidden, hidden, None).reshape(num_nonego, H, W, -1).permute(0,3,1,2)
                final_estimations = hidden.squeeze(1)                   # [non_ego_cars, C, H, W]
                if if_save_pt:
                    torch.save(final_estimations, '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ 'nonego_estimations'+'.pt')

                
                non_ego_mask_past_union = (torch.sum(non_ego_mask_past, dim=1)>0).to(torch.float32) # nonego_cavs, 1, H, W
                # torch.save(non_ego_mask_past_union,'/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ f'non_ego_mask_past_union_{self.dilation_size}' + '.pt')
                non_ego_mask_past_union = self.dilate_mask(non_ego_mask_past_union)
                if if_save_pt:
                    torch.save(non_ego_mask_past_union,'/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ f'non_ego_mask_past_union_dilated_{self.dilation_size}' + '.pt')

                final_estimations = final_estimations * non_ego_mask_past_union

                # debug use, uncomment to retrieve current mask
                ego_mask_curr = valid_mask[0]
                nonego_mask_curr = non_ego_mask[:, 0] # [nonego_cav, 1, H, W]
                # torch.save(nonego_mask_curr,'/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ f'nonego_mask_curr_{self.dilation_size}' + '.pt')

                individual_recon_loss_pos = torch.torch.nn.functional.smooth_l1_loss(final_estimations,nonego_splits[:,0],reduction='sum')/C
                if non_ego_mask_past_union.sum() != 0:
                    individual_recon_loss_pos /= (non_ego_mask_past_union.sum())

                # individual_recon_loss_neg = torch.torch.nn.functional.smooth_l1_loss((1-non_ego_mask_past_union)*final_estimations,nonego_splits[:,0],reduction='sum')
                # if (1-non_ego_mask_past_union).sum() != 0:
                    # individual_recon_loss_neg /= ((1-non_ego_mask_past_union).sum())

                # individual_recon_loss = individual_recon_loss_pos + individual_recon_loss_neg
                individual_recon_loss = individual_recon_loss_pos

                # latency_recon_loss = torch.torch.nn.functional.smooth_l1_loss(latency_features, nonego_splits[:,0],reduction='sum')/C
                # if non_ego_mask_past_union.sum() != 0:
                #     latency_recon_loss /= (non_ego_mask_past_union.sum())

                final_estimations = torch.cat((features[:1],final_estimations),dim=0)       # N, C, H, W
                # final_feature = torch.cat((features[:1],curr_features),dim=0)               # N, C, H, W
                # final_feature_latency = torch.cat((features[:1],latency_features),dim=0)    # N, C, H, W

                return final_estimations, individual_recon_loss #, latency_recon_loss

                # ###############skimage.dilation##########################################
                # final_fuse_estimation = torch.max(torch.cat((features[:1],final_estimations),dim=0), dim=0)[0]
                # final_fuse_feature = torch.max(torch.cat((features[:1],curr_features),dim=0),dim=0)[0]
                # # fuse_recon_loss = torch.torch.nn.functional.smooth_l1_loss(final_fuse_estimation,final_fuse_feature,reduction='mean')

                # final_fuse_feature_latency = torch.max(torch.cat((features[:1],latency_features),dim=0),dim=0)[0]
                # # ipdb.set_trace()
                # # print('#######################individual:',torch.torch.nn.functional.smooth_l1_loss(latest_features[1:],curr_features[1:]),individual_recon_loss)
                # # latest_features = features[self.sweep_length-1::self.sweep_length].clone()
                # # latest_features[0] = features[0]

                # # final1 = torch.max(torch.cat((features[:1],latest_features[1:]),dim=0), dim=0)[0]
                # # cnt = 5
                # # torch.save(curr_features,'/GPFS/data/yuxiwei/yuxiwei/OpenCOOD/vis/current'+str(cnt)+'.pt')
                # # torch.save(latest_features,'/GPFS/data/yuxiwei/yuxiwei/OpenCOOD/vis/latest'+str(cnt)+'.pt')
                # # cnt += 1
                # # ipdb.set_trace()
                # # print('#######################:fuse',torch.torch.nn.functional.smooth_l1_loss(final1,final_fuse_feature),fuse_recon_loss)
             
                # return final_fuse_estimation,final_fuse_feature,final_fuse_feature_latency, individual_recon_loss, latency_recon_loss # +fuse_recon_loss
                # # return final_fuse_feature, 0

            """ 
            num_cav = int(features.shape[0]/self.sweep_length)
            curr_features = features[:self.sweep_length:]
            features = torch.cat((features[:1],features[num_cav:]),dim=0)
            time_stamps = torch.cat((time_stamps[:1],time_stamps[num_cav:]),dim=0)
            cav_num, C, H, W = features.shape
            ################################
            # features = features[::2]
            # valid_mask = valid_mask[::2]
            # cav_num = int(cav_num/2)
            ################################


            if self.if_conv_aggre:
                curr = features[0:1].clone()
                prev = features[1:].clone()
                prev = self.conv_aggre(curr,prev)
                features = torch.cat((curr,prev),dim=0)
                features = features.permute(0,2,3,1)
                time_stamps = time_stamps*-1
                features = self.time_encoding(features,time_stamps)
                features = features.permute(0,3,1,2)
                if self.if_spatial_conv:
                    features = self.layers_spatial[0](features)
                return torch.max(features,dim=0)[0]
            
            if self.if_ego_time_encoding:
                # s1 = time.time()
                features = features.permute(0,2,3,1)
                time_stamps = time_stamps*-1
                features = self.time_encoding(features,time_stamps)
                features = features.permute(0,3,1,2)
                # s2 = time.time()
                # print('TE',s2-s1)
            
            
            features = features.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            # masks = valid_mask.view(cav_num, H*W).permute(1,0)
            # masks = masks.unsqueeze(-1).repeat(1,1,cav_num)
            


            if self.if_dotproductattn:
                features = self.dotproductattn(features, features, features, masks)
            if self.if_time_attn_aggre:
                # s1 = time.time()
                features = self.layers_temporal[0](features, features, features, None)
                # s2 = time.time()
                # print('trans',s2-s1)
            
           

            features = features.permute(1, 2, 0).view(cav_num, C, H, W)[:1] 
            if self.if_spatial_conv:
                # s1 = time.time()
                features = self.layers_spatial[0](features)[0]
                # s2 = time.time()
                # print('conv',s2-s1)
            if self.sup_individual:
                features = torch.max(final_estimations, dim=0)[0]
            curr_fuse_features = torch.max(curr_features, dim=0)[0]
            recon_loss = torch.torch.nn.functional.smooth_l1_loss(features,curr_fuse_features)

            return features, recon_loss
            """

        """
        valid_mask = valid_mask.squeeze(1)
        
        features = features.permute(0,2,3,1)
        
        features = torch.cat((features[0:1],features[self.sweep_length:]),dim=0)
        

        ##########################################  三种encoding
        if self.if_spatial_encoding:
            features[0:1] = self.spatial_encoding(features[0:1])# features.shape = [batch_all,height,width,feature_dim]
        if self.if_sensor_encoding and sensor_dist!=-1:
            sensor_dist = torch.cat((sensor_dist[0:1],sensor_dist[self.sweep_length:]))
            features = self.sensor_dist_encoding(features,sensor_dist)
        time_stamps = torch.cat((time_stamps[0:1],time_stamps[self.sweep_length:]),dim=0)
        valid_mask = torch.cat((valid_mask[0:1],valid_mask[self.sweep_length:]),dim=0)
        valid_mask_sum = valid_mask.sum(0)
        nonego_idx = torch.where(valid_mask_sum>1)
 
        features_to_aggre = features[:,nonego_idx[0],nonego_idx[1],:] 
        if self.if_ego_time_encoding:
            features[0:1] = self.time_encoding(features[0:1],time_stamps[0:1])

        if features_to_aggre.shape[1] != 0:
            if self.if_spatial_encoding:
                features_to_aggre = self.spatial_encoding_nonego(features_to_aggre,nonego_idx)
            if self.if_nonego_time_encoding:
                features_to_aggre = self.time_encoding(features_to_aggre,time_stamps)
        ###########################################

        ################################################### 
        # ipdb.set_trace()
        masks = valid_mask[:,nonego_idx[0],nonego_idx[1]]#(valid_num,len)
        batch_len = masks.shape[0]
        masks = masks.permute(1,0).unsqueeze(-1).repeat(1,1,batch_len)

        features_to_aggre = features_to_aggre.permute(1,0,2)
        for l in range(self.num_layers):
                
            if masks.shape[0] != 0:
                if self.if_dotproductattn:
                    # if self.use_mask == False:
                    #     masks =  None
                    # features_to_aggre = self.dotproductattn(features_to_aggre,features_to_aggre,features_to_aggre,masks)
                    features_to_aggre[:,0] = torch.max(features_to_aggre,dim=1)[0]
                else:
                    features_to_aggre = self.layers_temporal[l](features_to_aggre,features_to_aggre,features_to_aggre,masks)
                features[:,nonego_idx[0],nonego_idx[1],:] = features_to_aggre.permute(1,0,2)


            if self.if_spatial_conv:
                features = features.permute(0,3,1,2)
                features_backup = features.clone()
                features[0:1] = self.layers_spatial[l](features_backup[0:1])
                features = features.permute(0,2,3,1)

                features_to_aggre[:,0:1] = features[0:1,nonego_idx[0],nonego_idx[1],:].permute(1,0,2)

        return features[0].permute(2,0,1)
        """

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]
        
class raindrop_swin_w_single(nn.Module):
    def __init__(self, args):
        super(raindrop_swin_w_single, self).__init__()
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
        if self.agg_mode == 'RAIN':
            d_ff = args['d_ff']
            h = args['h']
            dropout = args['dropout']
            input_dim = args['input_dim']
            num_layers = args['num_layers']
            neighbor_range = args['neighbor_range']
            max_len = args['max_len']
            height = 100
            width = 352
            self.sweep_length = args['fusion_setting']['sweep_length']
            self.fuse_layer1 = False
            if self.multi_scale:
                layer_nums = args['layer_nums']
                num_filters = args['num_filters']
                self.num_levels = len(layer_nums)
                if self.fuse_layer1:
                    self.fuse_modules = AttentionFusion(args['fusion_setting'], input_dim, d_ff, h, num_layers, neighbor_range, height, width, dropout, max_len)
                    for p in self.fuse_modules.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                else:
                    self.fuse_modules = nn.ModuleList()
                    for idx in range(self.num_levels):
                        if idx==0:
                            fuse_network = AttentionFusion(args['fusion_setting'], input_dim, d_ff, h, num_layers, neighbor_range, height, width, dropout, max_len, compensation=True)
                        else: # TODO: 换一下！
                            fuse_network = AttentionFusion(args['fusion_setting'], input_dim, d_ff, h, num_layers, neighbor_range, height, width, dropout, max_len, compensation=False)
                        for p in fuse_network.parameters():
                            if p.dim() > 1:
                                nn.init.xavier_uniform_(p)

                        height = int(height/2)
                        width = int(width/2)
                        input_dim = input_dim * 2
                        d_ff = d_ff * 2
                        self.fuse_modules.append(fuse_network)
            else:
                input_dim = int(args['input_dim'])
                self.final_fuse = mask_fuse(input_dim)
                self.fuse_modules = AttentionFusion(args['fusion_setting'], input_dim, d_ff, h, num_layers, neighbor_range, height, width, dropout, max_len, compensation=True)
                # This was important from their code. 
                # Initialize parameters with Glorot / fan_avg.
                for p in self.fuse_modules.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

        else:
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

        print("##### Raindrop + Swin *with single supervision* Init! #####")
    
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
        cum_sum_len = torch.cumsum(len*k-(k-1), dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        
        return split_x

    def get_t_mat_for_past_1(self, all_mat):
        """
        all_mat:        
            [N*(k+1), 2, 3]

        Return:
        -------
        mat1: [N*(k+1), 2, 3]
        mat2: [N, 2, 3]
        """
        last_row = torch.tensor([[0,0,1]]).repeat(all_mat.shape[0], 1, 1).to(all_mat)

        all_mat_with_last_row = torch.cat((all_mat,last_row), dim=1).to(torch.float32)
        nonego_past1_to_egocurr = all_mat_with_last_row[1::self.sweep_length+1]
        # print(nonego_past1_to_egocurr.shape)
        nonego_past1_to_egocurr = nonego_past1_to_egocurr.to(torch.float32)
        egocurr_to_nonego_past1 = torch.inverse(nonego_past1_to_egocurr)
        egocurr_to_nonego_past1_repeat = torch.repeat_interleave(egocurr_to_nonego_past1, self.sweep_length+1, 0)
        # print(egocurr_to_nonego_past1_repeat.shape, all_mat_with_last_row.shape)
        all_to_nonego_past1 = torch.bmm(egocurr_to_nonego_past1_repeat, all_mat_with_last_row)

        return all_to_nonego_past1[:,:2], nonego_past1_to_egocurr[:,:2]

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
        # print('num',x.shape[0],pairwise_t_matrix.shape)
        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2
        
        if self.multi_scale:
            
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))
            
            if self.fuse_layer1:
                all_recon_loss = 0
                layer1_feats = backbone.resnet.forward_layer1(x)  # [B, C, H, W]
                if self.communication:
                    batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
                    batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...]
                    _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                    communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
                    # debug use, to save original feature:
                    # debug_feats = layer1_feats.
                    # TODO scale = 0 不加 mask
                    layer1_feats = layer1_feats * communication_masks_tensor

                batch_node_features = self.regroup(layer1_feats, record_len, K)
                x_fuse = []
                x_fuse_curr = []
                x_fuse_latency = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    # print('node and t mat',node_features.shape,t_matrix.shape,record_len)
                    t_matrix = t_matrix[K-1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix,
                                                    (H, W))
                    
                    # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                    record_frames = np.ones((N))*K; record_frames[0]-=(K-1)
                    if self.agg_mode == 'RAIN':
                        # for sensor embedding
                        sensor_dist = -1# (B, H, W)
                        # x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        features,features_curr,features_latency, recon_loss = self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b])
                        all_recon_loss += recon_loss # TODO: 是否需要添加 /= B ? 
                        x_fuse.append(features)
                        x_fuse_curr.append(features_curr)
                        x_fuse_latency.append(features_latency)
                x_fuse = torch.stack(x_fuse)  ###(batch, channel, h, w)
                x_fuse_curr = torch.stack(x_fuse_curr)
                x_fuse_latency = torch.stack(x_fuse_latency)

                layer2_feat, layer3_feat = backbone.resnet.forward_from_layer1(x_fuse)
                ups = []
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[0](x_fuse))
                    ups.append(backbone.deblocks[1](layer2_feat))
                    ups.append(backbone.deblocks[2](layer3_feat))
                
                if len(ups) > 1:
                    x_fuse = torch.cat(ups, dim=1)
                elif len(ups) == 1:
                    x_fuse = ups[0]
                
                if len(backbone.deblocks) > self.num_levels:
                    x_fuse = backbone.deblocks[-1](x_fuse)

                
                layer2_feat_curr, layer3_feat_curr = backbone.resnet.forward_from_layer1(x_fuse_curr)
                ups = []
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[0](x_fuse_curr))
                    ups.append(backbone.deblocks[1](layer2_feat_curr))
                    ups.append(backbone.deblocks[2](layer3_feat_curr))
                    
                if len(ups) > 1:
                    x_fuse_curr = torch.cat(ups, dim=1)
                elif len(ups) == 1:
                    x_fuse_curr = ups[0]
                
                if len(backbone.deblocks) > self.num_levels:
                    x_fuse_curr = backbone.deblocks[-1](x_fuse_curr)

                
                layer2_feat_latency, layer3_feat_latency = backbone.resnet.forward_from_layer1(x_fuse_latency)
                ups = []
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[0](x_fuse_latency))
                    ups.append(backbone.deblocks[1](layer2_feat_latency))
                    ups.append(backbone.deblocks[2](layer3_feat_latency))
                    
                if len(ups) > 1:
                    x_fuse_latency = torch.cat(ups, dim=1)
                elif len(ups) == 1:
                    x_fuse_latency = ups[0]
                
                if len(backbone.deblocks) > self.num_levels:
                    x_fuse_latency = backbone.deblocks[-1](x_fuse_latency)

            else:
                ups = []
                ups_curr = []
                ups_latency = []
                ups_single = []
                all_recon_loss = 0
                all_latency_recon_loss = 0
                for i in range(self.num_levels):
                    x = feats[i] if with_resnet else backbone.blocks[i](x)
                    ############ 1. Communication (Mask the features) #########
                    if i==0:
                        if self.communication:
                            batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
                            batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...]
                            _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                            communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
                            # for debug use
                            if if_save_pt:
                                x_wo_mask = x.clone()
                            x = x * communication_masks_tensor
                        else:
                            communication_rates = torch.tensor(0).to(x.device)
                    else:
                        if self.communication:
                            communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2)
                            # TODO scale = 1, 2 不加 mask
                            # x = x * communication_masks_tensor
                    
                    ############ 2. Split the confidence map #######################
                    # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                    # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                    # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                    # print('x',x.shape)
                    batch_node_features = self.regroup(x, record_len, K)
                    batch_comm_masks = self.regroup(communication_masks_tensor, record_len, K)
                    
                    if if_save_pt:
                        batch_x_wo_mask = self.regroup(x_wo_mask, record_len, K)
                    
                    ############ 3. Fusion ####################################
                    x_fuse = []
                    x_single = []
                    for b in range(B):
                        # number of valid agent
                        N = record_len[b]
                        # t_matrix[i, j]-> from i to j
                        t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) # [Nx(k+1), 2, 3]

                        mat_to_past1, mat_to_ego_curr = self.get_t_mat_for_past_1(t_matrix)
                        mat_to_past1 = mat_to_past1[K-1:]
                        node_features = batch_node_features[b]              # (N-1)xk+1, C, H, W
                        node_masks = batch_comm_masks[b]                    # (N-1)xk+1, 1, H, W
                        C, H, W = node_features.shape[1:]
                        if if_save_pt:
                            torch.save(batch_x_wo_mask[b], pt_save_path + 'feature_on_cav_past_0.pt')
                            torch.save(t_matrix[K-1:], pt_save_path + 't_matrix.pt')
                            torch.save(mat_to_past1, pt_save_path + 'mat_to_past1.pt')
                            torch.save(mat_to_ego_curr, pt_save_path + 't_matrmat_to_ego_curr.pt')
                            feature_on_ego_curr = warp_affine_simple(batch_x_wo_mask[b], t_matrix[K-1:], (H, W))
                            torch.save(feature_on_ego_curr, pt_save_path + 'feature_on_ego_curr.pt')
                        if self.agg_mode == 'RAIN':
                            # for sensor embedding
                            sensor_dist = -1# (B, H, W)
                            node_features = warp_affine_simple(node_features,
                                                        mat_to_past1,
                                                        (H, W))
                            node_masks = warp_affine_simple(node_masks,
                                                        mat_to_past1,
                                                        (H, W))
                            features, recon_loss = self.fuse_modules[i](node_features, sensor_dist, batch_time_intervals[b], node_masks)
                            x_single.append(features[1:])   # B x [n*, C, H, W]
                            all_recon_loss += recon_loss
                        else: # ATTEN, MAX, Transformer
                            x_fuse.append(self.fuse_modules[i](neighbor_feature))
                        
                        # print('node and t mat',node_features.shape,t_matrix.shape,record_len)
                        # t_matrix = t_matrix[K-1:]

                        neighbor_comp_feature = warp_affine_simple(features,
                                                        mat_to_ego_curr,
                                                        (H, W))
                        # neighbor_mask = warp_affine_simple(node_masks, mat_to_ego_curr, (H, W))

                        if if_save_pt:
                            torch.save(neighbor_comp_feature, pt_save_path+'feature_com'+'.pt')
                        # max fusion
                        neighbor_fused_feature = torch.max(neighbor_comp_feature, dim=0)[0]  # [C, H, W]
                        x_fuse.append(neighbor_fused_feature)

                        if if_save_pt:
                            neighbor_wo_mask = warp_affine_simple(batch_x_wo_mask[b], t_matrix, (H, W))
                            torch.save(neighbor_wo_mask[1:], '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/'+ 'feature_wo_mask'+'.pt')

                        # TODO: 这个位置
                        record_frames = np.ones((N))*K
                        # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                        
                        
                    x_single = torch.cat(x_single, dim=0) # [BN, C, H, W]
                    x_fuse = torch.stack(x_fuse)
                    ############ 4. Deconv ####################################
                    if len(backbone.deblocks) > 0:
                        ups.append(backbone.deblocks[i](x_fuse))
                        ups_single.append(backbone.deblocks[i](x_single))

                    else:
                        ups.append(x_fuse)
                    
                if len(ups) > 1:
                    x_fuse = torch.cat(ups, dim=1)
                elif len(ups) == 1:
                    x_fuse = ups[0]

                if len(ups_single) > 1:
                    x_single = torch.cat(ups_single, dim=1)
                elif len(ups_single) == 1:
                    x_single = ups_single[0]
                
                if len(backbone.deblocks) > self.num_levels:
                    x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 2. Communication (Mask the features) #########
            if self.communication:
                batch_confidence_maps = self.regroup(rm, record_len, K)
                # _, (B, 1, H, W), float
                _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
                x = x * communication_masks_tensor
            else:
                communication_rates = torch.tensor(0).to(x.device)

            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)
            batch_comm_masks = self.regroup(communication_masks_tensor, record_len, K)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            x_single = []
            all_recon_loss = 0.0
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                mat_to_past1, mat_to_ego_curr = self.get_t_mat_for_past_1(t_matrix)
                mat_to_past1 = mat_to_past1[K-1:]
                node_features = batch_node_features[b]
                node_masks = batch_comm_masks[b]                    # (N-1)xk+1, 1, H, W
                C, H, W = node_features.shape[1:]
                if if_save_pt:
                    torch.save(node_features, pt_save_path + 'feature_on_cav_past_0.pt')
                    torch.save(t_matrix[K-1:], pt_save_path + 't_matrix.pt')
                    torch.save(mat_to_past1, pt_save_path + 'mat_to_past1.pt')
                    torch.save(mat_to_ego_curr, pt_save_path + 't_matrmat_to_ego_curr.pt')
                    feature_on_ego_curr = warp_affine_simple(node_features, t_matrix[K-1:], (H, W))
                    torch.save(feature_on_ego_curr, pt_save_path + 'feature_on_ego_curr.pt')
                if self.agg_mode == 'RAIN':
                    sensor_dist = -1# (B, H, W)
                    node_features = warp_affine_simple(node_features,
                                                    mat_to_past1,
                                                    (H, W))
                    node_masks = warp_affine_simple(node_masks,
                                                        mat_to_past1,
                                                        (H, W))
                    features, recon_loss = self.fuse_modules(node_features, sensor_dist, batch_time_intervals[b], node_masks)
                    x_single.append(features[1:])   # B x [n*, C, H, W]
                    all_recon_loss += recon_loss
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
                
                neighbor_comp_feature = warp_affine_simple(features,
                                                        mat_to_ego_curr,
                                                        (H, W))

                # max fusion # TODO: max fusion 或 其他融合方法
                # neighbor_fused_feature = torch.max(neighbor_comp_feature, dim=0)[0]  # [C, H, W]
                neighbor_fused_feature = self.final_fuse(neighbor_comp_feature[0].unsqueeze(0), neighbor_comp_feature[1:])
                x_fuse.append(neighbor_fused_feature)
            x_fuse = torch.stack(x_fuse)
            x_single = torch.cat(x_single, dim=0) # [BN, C, H, W]
        
        return x_fuse, x_single, communication_rates, all_recon_loss, {}