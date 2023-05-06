# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F 
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.autograd import Variable
from opencood.utils import box_utils
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(vocab,d_model),nn.ReLU(),nn.Linear(d_model,d_model))

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding_irregular(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding_irregular, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model,d_model)
         
    def forward(self, x, time):
        ### x:[batch, agent, k, d] time:[batch, agent, k]
        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1])
            time = time.reshape(-1,time.shape[-1])

        pe = torch.zeros(x.shape[0],time.shape[1], self.d_model).cuda()
        position = time.unsqueeze(2) # 增加维度
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).cuda()#相对位置公式
         
        pe[:,:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:,:, 1::2] = torch.cos(position * div_term)   #取偶数列

        # x = x + Variable(pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加
        
        x = self.linear(torch.cat((x,Variable(pe[:, :x.size(1)], requires_grad=False)),dim=-1))
        x = x.reshape(last_shape)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.N = N

        self.norm = LayerNorm(layer.size) #归一化

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i in range(self.N):
            # print('x',x.shape)
            x = self.layers[i](x, mask) 

        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #sublayer 1 
        self.feed_forward = feed_forward #sublayer 2 
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #拷贝两个SublayerConnection，一个为了attention，一个是独自作为简单的神经网络
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # print('x',x.shape,'attn',self.self_attn(x,x,x,mask).shape)
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #attention层 对应层1
        return self.sublayer[1](x, self.feed_forward) # 对应 层2
        return x


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

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn #解码的attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, src_key,memory, tgt_mask):
        m = memory
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #self-attention
        x = self.sublayer[0](x, lambda x: self.src_attn(x, src_key, m, tgt_mask)) #解码
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # print('layersize',layer.size)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, src_key ,memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_key, memory, tgt_mask) #添加编码的后的结果
        return self.norm(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    # print('score',scores.shape,'mask',mask.shape)
    if mask is not None:
        mask = mask.transpose(1,2).cuda()
        scores = scores.masked_fill(mask == 0, -1e9) #mask必须是一个ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1 ，是将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value  ,mask value必须同为tensor
    
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
        self.d_k = d_model // h  # d_v=d_k=d_model/h 
        self.h = h # heads 的数目文中为8
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

        
        dim_to_keep = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query = self.linear_q(query).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        key = self.linear_k(key).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        value = self.linear_v(value).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,# 进行attention
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(2, 3).contiguous() \
            .view(nbatches, dim_to_keep ,-1, self.h * self.d_k) # 还原序列[batch_size,len,d_model]

        return self.linear_out(x)

class Generator(nn.Module):
    def __init__(self, d_model, out_dim):
        
        super(Generator, self).__init__()
        
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        
        return self.proj(x)


class Motion_prediction(nn.Module):
    def __init__(self, encoder, decoder, embedding_src, embedding_tgt, position_src, position_tgt, generator, embed_dim):
        super(Motion_prediction, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = embedding_src
        self.tgt_embed = embedding_tgt
        self.src_pe = position_src
        self.tgt_pe = position_tgt
        self.generator = generator

        

    def forward(self, input, query, time, future_time, mask=None):
        ################################ input: [batch, agent, past_frames, dim]  mask: [batch, agent, past_frames, past_frames]
        input = self.src_embed(input)
        input = self.src_pe(input, time)

        query = self.tgt_embed(query)
        query = self.tgt_pe(query, future_time)

        input_features = self.encoder(input, mask)
        
        prediction_features = self.decoder(query, input_features, input_features,mask)
        predictions = self.generator(prediction_features)
        return predictions



class Motion_interaction(nn.Module):
    def __init__(self, encoder, embedding, generator, neighbor_threshold):
        super(Motion_interaction, self).__init__()
        self.encoder = encoder
        self.src_embed = embedding
        self.generator = generator
        self.neighbor_threshold = neighbor_threshold


    def forward(self, input):
        ####input:[batch,agent,2]
        mask = torch.cdist(input,input)

        mask = torch.where(mask<self.neighbor_threshold,1,0)
        input = input.unsqueeze(1)
        mask = mask.unsqueeze(1)

        input = self.src_embed(input)

        input_features = self.encoder(input, mask)
        predictions = self.generator(input_features)

        return predictions


def make_model(input_dim, output_dim, num_layers=2, d_model=64, d_ff=128, num_heads=2, dropout=0.1,neighbor_shreshold=10):
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, d_model,d_model,d_model,d_model)
    attn_decoder = MultiHeadedAttention(num_heads,d_model,d_model,d_model,d_model)
    ff = PositionwiseFeedForward(d_model, d_model, d_ff,dropout)
    position = PositionalEncoding_irregular(d_model, dropout)

    model_prediction = Motion_prediction(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, input_dim),
        Embeddings(d_model, output_dim), 
        c(position),
        c(position),
        Generator(d_model, output_dim),
        d_model)

    model_interaction = Motion_interaction(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, input_dim), 
        Generator(d_model, output_dim),
        neighbor_shreshold)
    
    for p in model_prediction.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in model_interaction.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model_prediction, model_interaction

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

'''
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
'''

class Matcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, fusion, cost_dist: float = 1, cost_giou: float = 1, thre: float = 20):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_dist = cost_dist
        self.cost_giou = cost_giou
        self.thre = thre

        self.fusion = fusion
        if fusion=='flow':
            m1, m2 = make_model(4,2)
            self.compensate_motion = m1

    @torch.no_grad()
    def forward(self, input_dict, feature=None, shape_list=None, batch_id=0, viz_flag=False):
        self.viz_flag = viz_flag
        if self.fusion=='box':
            return self.forward_box(input_dict, batch_id)
        elif self.fusion=='feature':
            return self.forward_feature(input_dict, feature)
        # elif self.fusion=='flow':
        #     return self.forward_flow_multi_frames(input_dict, shape_list)
        elif self.fusion=='linear':
            return self.forward_flow(input_dict, shape_list)
        elif self.fusion=='flow': # TODO: flow_dir
            return self.forward_flow_dir(input_dict, shape_list)
        else:
            print("Attention, fusion method must be in box or feature!")

    def forward_box(self, input_dict, batch_id):
        """ Performs the matching
        input_dict: 
            {
                'ego' : {
                    'past_k_time_diff' : 
                    [0] {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                    ... 
                    [k-1]
                    ['comp']{
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
                cav_id {}
            }
        """

        # output_dict_past = {}
        # output_dict_current = {}
        pred_bbox_list = []
        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                estimated_box_center_current = cav_content[0]['pred_box_center_tensor']
                estimated_box_3dcorner_current = cav_content[0]['pred_box_3dcorner_tensor']
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]
                
                # TODO: norm
                # center_points_past1_norm = center_points_past1 - torch.mean(center_points_past1,dim=0,keepdim=True)
                # center_points_past2_norm = center_points_past2 - torch.mean(center_points_past2,dim=0,keepdim=True)
                # cost_mat_center = torch.cdist(center_points_past2_norm,center_points_past1_norm) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                # if flow.shape[0] != 0:
                #     print(f"max flow is {flow.max()}")

                estimate_position = matched_past1 + flow*(0-cav_content['past_k_time_diff'][0]) 

                # from copy import deepcopy
                # estimated_box_center_current = deepcopy(coord_past1['pred_box_center_tensor'].detach())
                estimated_box_center_current = coord_past1['pred_box_center_tensor'].detach().clone()
                estimated_box_center_current[past1_ids, :2] = estimate_position  # n, 7

                # estimated_box_center_current = torch.zeros_like(coord_past1['pred_box_center_tensor']).to(estimate_position.device)
                # estimated_box_center_current[past1_ids] += torch.cat([estimate_position, coord_past1['pred_box_center_tensor'][past1_ids][:,2:]], dim=-1)
                # no_past1_ids = [x for x in range(coord_past1['pred_box_center_tensor'].shape[0]) if x not in list(past1_ids)]
                # estimated_box_center_current[no_past1_ids] += coord_past1['pred_box_center_tensor'][no_past1_ids]

                estimated_box_3dcorner_current = box_utils.boxes_to_corners_3d(estimated_box_center_current, order='hwl')

            # debug use, update input dict adding estimated frame at cav-past0
            input_dict[cav]['comp'] = {}
            input_dict[cav]['comp'].update({
                'pred_box_center_tensor': estimated_box_center_current,
                'pred_box_3dcorner_tensor': estimated_box_3dcorner_current,
                'scores': cav_content[0]['scores']
            })

        return input_dict

    def forward_feature(self, input_dict, features_dict):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                'ego' / cav_id : {
                    [0] / [1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        pred_bbox_list = []
        
        ######## TODO: debug use
        # # pop up other cavs, only keep ego:
        # debug_dict = {'ego': input_dict['ego']}
        # input_dict = debug_dict.copy()
        # debug_features_dict = {'ego': features_dict['ego']}
        # features_dict = debug_features_dict.copy()
        ##############

        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                updated_spatial_features_2d = features_dict[cav]['spatial_features_2d'][0]
                updated_spatial_features = features_dict[cav]['spatial_features'][0]
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                # TODO: flow * (0-past_k[0])
                flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict

    def forward_flow_multi_frames(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items():
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list)
                flow_map_list.append(basic_warp_mat)
                reserved_mask.append(mask)
            else:
                past_k_time_diff = cav_content['past_k_time_diff']

                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]
                coord_past3 = cav_content[2]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]
                center_points_past3 = coord_past3['pred_box_center_tensor'][:,:2]

                self.thre_post_process = 10

                # past1 and past2 match
                cost_mat_center_a = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]
                # original_cost_mat_center_a = cost_mat_center_a.clone()
                # cost_mat_center_a[cost_mat_center_a > self.thre_post_process] = 1000

                cost_mat_center_drop_2_a = torch.sum(torch.where(cost_mat_center_a > self.thre, 1, 0), dim=1)
                dist_valid_past2_a = torch.where(cost_mat_center_drop_2_a < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center_a > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center_a = cost_mat_center_a[dist_valid_past2_a[0], :]
                cost_mat_center_a = cost_mat_center_a[:, dist_valid_past1[0]]
                
                cost_mat = cost_mat_center_a.clone()
                past2_ids_a, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids_a = dist_valid_past2_a[0][past2_ids_a]
                past1_ids = dist_valid_past1[0][past1_ids]

                if len(past2_ids_a)==0:
                    print('======= No matched boxes between latest 2 frames! =======')
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list) # TODO: rethink ones or zeros
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                    continue

                # past2 and past3 match
                cost_mat_center_b = torch.cdist(center_points_past3, center_points_past2) # [num_cav_past3,num_cav_past2]

                # original_cost_mat_center_b = cost_mat_center_b.clone()
                # cost_mat_center_b[cost_mat_center_b > self.thre_post_process] = 1000

                cost_mat_center_drop_3 = torch.sum(torch.where(cost_mat_center_b > self.thre, 1, 0), dim=1)
                dist_valid_past3 = torch.where(cost_mat_center_drop_3 < center_points_past2.shape[0])
                cost_mat_center_drop_2_b = torch.sum(torch.where(cost_mat_center_b > self.thre, 1, 0), dim=0)
                dist_valid_past2_b = torch.where(cost_mat_center_drop_2_b < center_points_past3.shape[0])

                cost_mat_center_b = cost_mat_center_b[dist_valid_past3[0], :]
                cost_mat_center_b = cost_mat_center_b[:, dist_valid_past2_b[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center_b.clone()
                past3_ids, past2_ids_b = linear_sum_assignment(cost_mat.cpu())
                
                past3_ids = dist_valid_past3[0][past3_ids]
                past2_ids_b = dist_valid_past2_b[0][past2_ids_b]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                # find the matched obj among three frames
                a_idx, b_idx = self.get_common_elements(past2_ids_a, past2_ids_b)

                # 最近两帧有匹配的，但是最近的三帧没有
                # there is no matched object in past frames
                if len(a_idx)==0 or len(b_idx)==0:
                    matched_past2 = center_points_past2[past2_ids_a]
                    matched_past1 = center_points_past1[past1_ids]

                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    flow = (matched_past1 - matched_past2) / time_length

                    flow = flow*(0-cav_content['past_k_time_diff'][0])
                    selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                    selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    flow_map_list.append(flow_map)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([past1_ids, past2_ids_a], dim=1))
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                        compensated_results_list.append(selected_box_3dcorner_compensated)
                    continue

                # 三帧匹配结果输入预测模块
                matched_past1 = center_points_past1[past1_ids[a_idx]].unsqueeze(0)
                matched_past2 = center_points_past2[past2_ids_a[a_idx]].unsqueeze(0)
                matched_past3 = center_points_past3[past3_ids[b_idx]].unsqueeze(0)

                obj_coords = torch.cat([matched_past3, matched_past2, matched_past1], dim=0)
                obj_coords = obj_coords.permute(1, 0, 2) # (N, k, 2)

                obj_coords_norm = obj_coords - obj_coords[:, -1:, :] # (N, k, 2)
                past_k_time_diff = torch.flip(past_k_time_diff, dims=[0]) # (k,) TODO: check if this is correct
                past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[-1] # (k,)

                speed = torch.zeros_like(obj_coords_norm) # (N, k, 2)
                speed[:, 1:, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), ((past_k_time_diff[1:] - past_k_time_diff[:-1]).unsqueeze(-1)).unsqueeze(0)) # (N, k-1, 2) / (1, k-1, 1)    

                obj_input = torch.cat([obj_coords_norm, speed], dim=-1) # (N, k, 4)

                obj_input = obj_input.unsqueeze(0) # (1, N, k, 4)
                
                last_time_length = (past_k_time_diff_norm[-1] - past_k_time_diff_norm[-2])
                if last_time_length == 0:
                    print("==== Warning! You met repeated package! ====")
                    query = torch.zeros(obj_input.shape)[:,:,:1,:2].to(obj_input.device) # (1, N, 1, 2)
                else: 
                    query = obj_coords_norm[:, -1:, :] + \
                        (obj_coords_norm[:, -1:, :]-obj_coords_norm[:, -2:-1, :])*(0-past_k_time_diff[-1]) / \
                            last_time_length
                    query = query.unsqueeze(0) # (1, N, 1, 2)

                target_time_diff = torch.tensor([-past_k_time_diff[-1]]).to(obj_input.device) # (1,)

                # target_time_diff = torch.tensor([-past_k_time_diff[0]]).to(obj_input.device) # (1,)
                compensated_coords_norm = self.compensate_motion(obj_input, query, past_k_time_diff_norm, target_time_diff) + query

                flow = compensated_coords_norm.squeeze(0).squeeze(1) # (N, 2)

                # flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids[a_idx],]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')

                if self.viz_flag and not(len(a_idx) < len(past2_ids_a)):
                    unit_matched_list = torch.stack([past1_ids[a_idx], past2_ids_a[a_idx], past3_ids[b_idx]], dim=1) # (N_obj, 3)
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 

                # 两帧匹配成功 but三帧匹配失败 将两帧的结果进行插值 com 表示补集
                if len(a_idx) < len(past2_ids_a): 
                    com_past1_ids = [elem.item() for id, elem in enumerate(past1_ids) if id not in a_idx]
                    com_past2_ids = [elem.item() for id, elem in enumerate(past2_ids_a) if id not in a_idx]
                    matched_past1 = center_points_past1[com_past1_ids]
                    matched_past2 = center_points_past2[com_past2_ids]
                    
                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    com_flow = (matched_past1 - matched_past2) / time_length

                    com_flow = com_flow*(0-cav_content['past_k_time_diff'][0])
                    com_selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][com_past1_ids,]
                    com_selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(com_selected_box_3dcenter_past0, order='hwl')
                    
                    flow = torch.cat([flow, com_flow], dim=0)
                    selected_box_3dcorner_past0 = torch.cat([selected_box_3dcorner_past0, com_selected_box_3dcorner_past0], dim=0)

                    # matched: 
                    # past1: past1_ids[a_idx] + com_past1_ids
                    # past2: past2_ids_a[a_idx] + com_past2_ids
                    # past3: past3_ids[b_idx]
                    if self.viz_flag:
                        tmp_past_1 = torch.cat([past1_ids[a_idx], torch.tensor(com_past1_ids).to(past1_ids)], dim=0)
                        tmp_past_2 = torch.cat([past2_ids_a[a_idx], torch.tensor(com_past2_ids).to(past1_ids)], dim=0)
                        unit_matched_list = torch.stack([tmp_past_1, tmp_past_2], dim=1)  # (N_obj, 2)
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 

                if self.viz_flag:
                    matched_idx_list.append(unit_matched_list)
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]
        
        if self.viz_flag:
            return final_flow_map, reserved_mask, matched_idx_list, compensated_results_list
        
        return final_flow_map, reserved_mask

    def forward_flow(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            matched_idx_list = []
        for cav, cav_content in input_dict.items():
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list)
                flow_map_list.append(basic_warp_mat)
                reserved_mask.append(mask)
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                self.thre_post_process = 10
                original_cost_mat_center = cost_mat_center.clone()
                cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                
                ### a trick
                matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                past2_ids = past2_ids[valid_mat_idx[0]]
                past1_ids = past1_ids[valid_mat_idx[0]]
                ####################

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            return final_flow_map, reserved_mask, matched_idx_list
        return final_flow_map, reserved_mask
        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''

    def forward_flow_dir(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items():
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list)
                flow_map_list.append(basic_warp_mat)
                reserved_mask.append(mask)
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device)

                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1)
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1]
                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495), 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat)

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre)

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0:
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                    continue

                match = torch.min(cost_mat_center, dim=1)
                match_to_keep = torch.where(match[0] < 5)

                past2_ids = match_to_keep[0]
                past1_ids = match[1][match_to_keep[0]]

                # ##############################################
                # self.thre_post_process = 10
                # original_cost_mat_center = cost_mat_center.clone()
                # cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                # cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                # dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                # cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                # dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                # cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                # cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # # cost_mat_iou = get_ious()
                # cost_mat = cost_mat_center
                # past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                # past2_ids = dist_valid_past2[0][past2_ids]
                # past1_ids = dist_valid_past1[0][past1_ids]
                
                # ### a trick
                # matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                # valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                # past2_ids = past2_ids[valid_mat_idx[0]]
                # past1_ids = past1_ids[valid_mat_idx[0]]
                # ####################

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            return final_flow_map, reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask
        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''

    def generate_flow_map(self, flow, bbox_list, scale=1.25, shape_list=None, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale
        flow:[num_cav, 2] at cav coodinate system and lidar scale
            bbox_list & flow : x and y are exactly image coordinate
            ------------> x
            |
            |
            |
            y
        scale: float, scale meters to voxel, feature_length / lidar_range_length = 1.25 or 2.5

        Returns
        -------
        updated_feature: feature after being warped by flow, [C, H, W]
        """
        # flow = torch.tensor([70, 0]).unsqueeze(0).to(feature)

        # only use x and y
        bbox_list = bbox_list[:, :, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

        flag_viz = False
        #######
        # store two parts of bbx: 1. original bbx, 2. 
        if flag_viz:
            viz_bbx_list = bbox_list
            fig, ax = plt.subplots(4, 1, figsize=(5,11))
            ######## viz-0: original feature, original bbx
            canvas_ori = viz_on_canvas(feature, bbox_list, scale=scale)
            plt.sca(ax[0])
            # plt.axis("off")
            plt.imshow(canvas_ori.canvas)
            ##########
        #######

        C, H, W = shape_list
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device)
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # C, H, W 
        if flow.shape[0] == 0 : 
            # reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            return basic_warp_mat,  reserved_area.unsqueeze(0)  # 返回不变的矩阵

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(shape_list.device) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
        ######### viz-1: original feature, original bbx and flowed bbx
        if flag_viz:
            viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
            canvas_hidden = viz_on_canvas(feature, viz_bbx_list, scale=scale)
            plt.sca(ax[1])
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        # generate mask
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1

        return basic_warp_mat, reserved_area.unsqueeze(0)
        
        '''
        ##################################### below is not used
        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        if flag_viz:
            p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
            p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
            p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
            p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
            warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
            viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
            canvas_new = viz_on_canvas(final_feature, viz_bbx_list, scale=scale)
            plt.sca(ax[2]) 
            # plt.axis("off") 
            plt.imshow(canvas_new.canvas)
        ############## 

        reserved_area = torch.ones_like(feature)  # C, H, W
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        final_feature = final_feature * reserved_area

        ####### viz-3: mask area out of warped bbx
        if flag_viz:
            partial_feature_one = torch.zeros_like(feature)  # C, H, W
            for cav in range(num_cav):
                partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
            masked_final_feature = partial_feature_one * final_feature
            canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list, scale=scale)
            plt.sca(ax[3]) 
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##############

        ####### viz: draw figures
        if flag_viz:
            plt.tight_layout()
            plt.savefig(f'result_canvas_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()

            fig, axes = plt.subplots(2, 1, figsize=(4, 4))
            major_ticks_x = np.linspace(0,350,8)
            minor_ticks_x = np.linspace(0,350,15)
            major_ticks_y = np.linspace(0,100,3)
            minor_ticks_y = np.linspace(0,100,5)
            for i, ax in enumerate(axes):
                plt.sca(ax); #plt.axis("off")
                ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
                ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
                ax.grid(which='major', color='w', linewidth=0.4)
                ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
                if i==0:
                    plt.imshow(torch.max(feature, dim=0)[0].cpu())
                else:
                    plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
            plt.tight_layout()
            plt.savefig(f'result_features_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()
        #######

        return final_feature
        '''

    def feature_warp(self, feature, bbox_list, flow, scale=1.25, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale
        flow:[num_cav, 2] at cav coodinate system and lidar scale
            bbox_list & flow : x and y are exactly image coordinate
            ------------> x
            |
            |
            |
            y
        scale: float, scale meters to voxel, feature_length / lidar_range_length = 1.25 or 2.5

        Returns
        -------
        updated_feature: feature after being warped by flow, [C, H, W]
        """
        # flow = torch.tensor([70, 0]).unsqueeze(0).to(feature)

        if flow.shape[0] == 0 : 
            return feature

        # only use x and y
        bbox_list = bbox_list[:, :, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

        flag_viz = False
        #######
        # store two parts of bbx: 1. original bbx, 2. 
        if flag_viz:
            viz_bbx_list = bbox_list
            fig, ax = plt.subplots(4, 1, figsize=(5,11))
            ######## viz-0: original feature, original bbx
            canvas_ori = viz_on_canvas(feature, bbox_list, scale=scale)
            plt.sca(ax[0])
            # plt.axis("off")
            plt.imshow(canvas_ori.canvas)
            ##########
        #######

        C, H, W = feature.size()
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([feature.shape[2], feature.shape[1]]).to(feature)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(feature) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
        ######### viz-1: original feature, original bbx and flowed bbx
        if flag_viz:
            viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
            canvas_hidden = viz_on_canvas(feature, viz_bbx_list, scale=scale)
            plt.sca(ax[1])
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        if flag_viz:
            p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
            p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
            p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
            p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
            warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
            viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
            canvas_new = viz_on_canvas(final_feature, viz_bbx_list, scale=scale)
            plt.sca(ax[2]) 
            # plt.axis("off") 
            plt.imshow(canvas_new.canvas)
        ############## 

        reserved_area = torch.ones_like(feature)  # C, H, W
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        final_feature = final_feature * reserved_area

        ####### viz-3: mask area out of warped bbx
        if flag_viz:
            partial_feature_one = torch.zeros_like(feature)  # C, H, W
            for cav in range(num_cav):
                partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
            masked_final_feature = partial_feature_one * final_feature
            canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list, scale=scale)
            plt.sca(ax[3]) 
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##############

        ####### viz: draw figures
        if flag_viz:
            plt.tight_layout()
            plt.savefig(f'result_canvas_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()

            fig, axes = plt.subplots(2, 1, figsize=(4, 4))
            major_ticks_x = np.linspace(0,350,8)
            minor_ticks_x = np.linspace(0,350,15)
            major_ticks_y = np.linspace(0,100,3)
            minor_ticks_y = np.linspace(0,100,5)
            for i, ax in enumerate(axes):
                plt.sca(ax); #plt.axis("off")
                ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
                ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
                ax.grid(which='major', color='w', linewidth=0.4)
                ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
                if i==0:
                    plt.imshow(torch.max(feature, dim=0)[0].cpu())
                else:
                    plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
            plt.tight_layout()
            plt.savefig(f'result_features_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()
        #######

        return final_feature

    def backup_feature_warp(self, feature, bbox_list, flow, scale=1.25, align_corners=False):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale
        flow:[num_cav, 2] at cav coodinate system and lidar scale
            bbox_list & flow : x and y are exactly image coordinate
            ------------> x
            |
            |
            |
            y
        scale: float, scale meters to voxel, feature_length / lidar_range_length = 1.25 or 2.5

        Returns
        -------
        updated_feature: feature after being warped by flow, [C, H, W]
        """
        # flow = torch.tensor([70, 0]).unsqueeze(0).to(feature)

        if flow.shape[0] == 0 : 
            return feature

        # only use x and y
        bbox_list = bbox_list[:, :, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

        # # store two parts of bbx: 1. original bbx, 2. 
        # viz_bbx_list = bbox_list
        # fig, ax = plt.subplots(4, 1, figsize=(5,11))
        
        # ######## viz-0: original feature, original bbx
        # canvas_ori = viz_on_canvas(feature, bbox_list)
        # plt.sca(ax[0])
        # # plt.axis("off")
        # plt.imshow(canvas_ori.canvas)
        # ##########

        C, H, W = feature.size()
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([feature.shape[2], feature.shape[1]]).to(feature)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(feature) # .to() 统一数据格式 float32
        
        ######### viz-1: original feature, original bbx and flowed bbx
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
        # viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
        # canvas_hidden = viz_on_canvas(feature, viz_bbx_list)
        # plt.sca(ax[1])
        # # plt.axis("off") 
        # plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + 176).to(torch.int) # TODO: 这里面的176需要重新考虑
        x_max_fid = (x_max + 176).to(torch.int)
        y_min_fid = (y_min + 50).to(torch.int)  # TODO: 这里面的50需要重新考虑
        y_max_fid = (y_max + 50).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        # p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
        # p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
        # p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
        # p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
        # warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
        # viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
        # canvas_new = viz_on_canvas(final_feature, viz_bbx_list)
        # plt.sca(ax[2]) 
        # # plt.axis("off") 
        # plt.imshow(canvas_new.canvas)
        ############## 

        ####### viz-3: mask area out of warped bbx
        # partial_feature_one = torch.zeros_like(feature)  # C, H, W
        # for cav in range(num_cav):
        #     partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        # masked_final_feature = partial_feature_one * final_feature
        # canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list)
        # plt.sca(ax[3]) 
        # # plt.axis("off") 
        # plt.imshow(canvas_hidden.canvas)
        ##############

        # plt.tight_layout()
        # plt.savefig('result_canvas.jpg', transparent=False, dpi=400)
        # plt.clf()

        # fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        # major_ticks_x = np.linspace(0,350,8)
        # minor_ticks_x = np.linspace(0,350,15)
        # major_ticks_y = np.linspace(0,100,3)
        # minor_ticks_y = np.linspace(0,100,5)
        # for i, ax in enumerate(axes):
        #     plt.sca(ax); #plt.axis("off")
        #     ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
        #     ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
        #     ax.grid(which='major', color='w', linewidth=0.4)
        #     ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
        #     if i==0:
        #         plt.imshow(torch.max(feature, dim=0)[0].cpu())
        #     else:
        #         plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
        # plt.tight_layout()
        # plt.savefig('result_features.jpg', transparent=False, dpi=400)
        # plt.clf()

        return final_feature

    def get_common_elements(self, A, B):
        common_elements_A = []
        common_elements_B = []
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                if a == b:
                    common_elements_A.append(i)
                    common_elements_B.append(j)
        return common_elements_A, common_elements_B


def get_center_points(corner_points):
    corner_points2d = corner_points[:,:4,:2]

    centers_x = torch.mean(corner_points2d[:,:,0],dim=1,keepdim=True)

    centers_y = torch.mean(corner_points2d[:,:,1],dim=1,keepdim=True)

    return torch.cat((centers_x,centers_y), dim=1)

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)