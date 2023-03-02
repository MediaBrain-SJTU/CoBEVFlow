# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.tools.debug_tools import viz_compensation_latefusion_flow
def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor

def inference_late_fusion_flow(batch_data, model, dataset, batch_id = 0):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    thre = 20
    delay = 300

    folder_name=f'viz_wo_norm_thre_{thre}_delay_{delay}'

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    box_results = dataset.generate_pred_bbx_frames(batch_data, output_dict)
    if batch_id%10 == 0:
        viz_compensation_latefusion_flow(dataset, batch_data, box_results, file_name=folder_name, save_notes='no_comp', vis_comp_box=False, batch_id=batch_id)

    ''' 
    box_result : dict for each cav at each time
        {
            'ego' / cav_id : {
                'past_k_time_diff' : 
                [0] ... [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                }
            }
        }
    ''' 
    
    updated_box = box_flow_update(box_results)
    '''
    updated_box: dict {
        'past_k_time_diff' : 
        [0], [1], ... , [k-1]
        'comp' : {
            pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3  (cav-past0)
            pred_box_center_tensor : n, 7  (cav-past0)
            scores: (n, )  
        }
    } 
    '''

    # 可视化每一帧的 box frame 与 final GT 进行对比
    if batch_id%10 == 0:
        viz_compensation_latefusion_flow(dataset, batch_data, updated_box, file_name=folder_name, save_notes='w_comp', vis_comp_box=True, batch_id=batch_id)

    # return batch_id, batch_id, batch_id

    # TODO: warp each cavs' estimated frame to ego (cav-past-0 to ego-curr)
    pred_box_tensor, pred_score, delay_box_tensor = dataset.post_process_updated(batch_data, updated_box)
    gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
    
    return pred_box_tensor, pred_score, gt_box_tensor, delay_box_tensor

def box_flow_update(box_results):
    """
    Calculate flow using the detection results of two adjacent frames, 
    and update the detection results using the flow.

    Parameters
    ----------
    box_results: dict

    Return 
    pred_box_tensor: torch.Tensor, boxes after late fusion. N_pred, 8, 3 
    pred_score: torch.Tensor, scores after NMS filter
        
    """
    from opencood.tools.matcher import Matcher
    matcher = Matcher(1, 1)
    updated_box = matcher(box_results)

    return updated_box

def inference_intermediate_fusion_flow(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    thre = 20
    delay = 300

    folder_name=f'viz_wo_norm_thre_{thre}_delay_{delay}_inte_flow'

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content) # point_pillar_w_flow

    # 要对 output_dict 里面的 spatial_feature_2d 进行 flow update
    # step1: 生成每一帧的 box 结果
    box_results = dataset.generate_pred_bbx_frames(batch_data, output_dict)
    # step2: a. 根据检测结果生成 flow ; b. 更新 spatial feature  :  'updated_spatial_feature_2d' # TODO: 这里面添加一个可视化
    updated_output_dict = feature_flow_update(output_dict, box_results)

    ####### debug use, viz updated feature of each cav
    # from matplotlib import pyplot as plt
    # viz_save_path = '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/debug_4_feature_flow'
    # for cav_id, cav_content in updated_output_dict.items():
    #     viz_content = torch.max(cav_content['updated_spatial_feature_2d'], dim=0)[0].detach().cpu()
    #     plt.imshow(viz_content)
    #     plt.savefig(viz_save_path+f'/updated_feature_{cav_id}.png')
    ##############
    
    # step3: 根据 spatial feature, 融合到 ego-curr
    # step4: detection header 
    pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']
    fused_dict = model(updated_output_dict, stage=2, pairwise_t_matrix=pairwise_t_matrix)
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_for_intermediate(batch_data, fused_dict)
    
    return pred_box_tensor, pred_score, gt_box_tensor

    # return inference_early_fusion(batch_data, model, dataset)

def feature_flow_update(output_dict, box_results):
    """
    generate box flow, and use the flow to update the feature 
    """
    # step 1. use box_results to generate flow
    # step 2. use flow to update spatial feature
    from opencood.tools.matcher import Matcher
    matcher = Matcher(1, 1, thre=20)
    updated_output_dict = matcher(box_results, fusion='feature', feature=output_dict)
    return updated_output_dict

def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return pred_box_tensor, pred_score, gt_box_tensor

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion(batch_data, # only for late fusion dataset
                             output_dict_ego, return_uncertainty=True)

    return pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    # return_dict = {"pred_box_tensor" : pred_box_tensor, \
    #                     "pred_score" : pred_score, \
    #                     "gt_box_tensor" : gt_box_tensor}

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)
