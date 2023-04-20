# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F 
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
    def forward(self, input_dict, fusion='box', feature=None, shape_list=None, batch_id=0):
        if fusion=='box':
            return self.forward_box(input_dict, batch_id)
        elif fusion=='feature':
            return self.forward_feature(input_dict, feature)
        elif fusion=='flow':
            return self.forward_flow(input_dict, shape_list)
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

    def forward_flow(self, input_dict, shape_list):
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
        flow_map_list = []
        reserved_mask = []
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
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]
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
        reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W
        if flow.shape[0] == 0 : 
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
        
        # below is not used
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

def get_center_points(corner_points):
    corner_points2d = corner_points[:,:4,:2]

    centers_x = torch.mean(corner_points2d[:,:,0],dim=1,keepdim=True)

    centers_y = torch.mean(corner_points2d[:,:,1],dim=1,keepdim=True)

    return torch.cat((centers_x,centers_y), dim=1)

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)