# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from opencood.utils import box_utils

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

    def __init__(self, cost_dist: float = 1, cost_giou: float = 1, thre: float = 20):
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

    @torch.no_grad()
    def forward(self, input_dict, fusion='box', feature=None):
        if fusion=='box':
            return self.forward_box(input_dict)
        elif fusion=='feature':
            return self.forward_feature(input_dict, feature)
        else:
            print("Attention, fusion method must be in box or feature!")

    def forward_box(self, input_dict):
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
                    'spatial_feature_2d'
                }
            }
        """
        pred_bbox_list = []
        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                updated_spatial_feature_2d = features_dict[cav]['spatial_features_2d'][0]
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

                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(matched_past1, order='hwl')
                updated_spatial_feature_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow)

            features_dict[cav].update({
                'updated_spatial_feature_2d': updated_spatial_feature_2d
            })

        return features_dict

    def feature_warp(self, feature, bbox_list, flow):
        """
        feature:[C, H, W]
        bbox_list:[num_cav, 4, 2]
        flow:[num_cav, 2]
        """
        return feature
        C, H, W = feature.size()
        num_cav = bbox_list.shape[0]
        # grid = F.affine_grid(M,
        #                      [B, C, dsize[0], dsize[1]],
        #                      align_corners=align_corners).to(src) # .to() 统一数据格式 float32
        basic_mat = torch.tensor([[1,0,0],[0,1,0]])
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

        cav_t_mat = []########### get_from_flow       shape:[num_cav,2,3]        affine matrix

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(src) # .to() 统一数据格式 float32

        x_min = torch.min(bbox_list[:,:,0],dim=1)[0] + flow[:,:0]
        x_max = torch.max(bbox_list[:,:,0],dim=1)[0] + flow[:,:0]

        y_min = torch.min(bbox_list[:,:,1],dim=1)[0] + flow[:,:1]
        y_max = torch.max(bbox_list[:,:,1],dim=1)[0] + flow[:,:1]

        for cav in range(num_cav):
            basic_warp_mat[0,x_min[cav]:x_max[cav],y_min[cav]:y_max[cav]] = cav_warp_mat[cav,x_min[cav]:x_max[cav],y_min[cav]:y_max[cav]]

        final_feature = F.grid_sample(feature, basic_warp_mat, align_corners=align_corners)

        return final_feature
    
def get_center_points(corner_points):
    corner_points2d = corner_points[:,:4,:2]

    centers_x = torch.mean(corner_points2d[:,:,0],dim=1,keepdim=True)

    centers_y = torch.mean(corner_points2d[:,:,1],dim=1,keepdim=True)

    return torch.cat((centers_x,centers_y), dim=1)

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)