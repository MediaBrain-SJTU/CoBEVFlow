# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import sys
from time import time
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils, simple_vis
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_delay import IntermediateFusionDatasetDAIRAsync
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization_dair.yaml'))
    time_delay = params['time_delay']
    output_path = "/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/debug"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    opencda_dataset = IntermediateFusionDatasetDAIRAsync(params, visualize=True,
                                            train=False)
    # dataset_len = len(opencda_dataset)
    # # print(dataset_len)
    # sampled_indices = np.random.permutation(dataset_len)[:2]

    sampled_indices = np.array(range(15))
    print(sampled_indices)
    
    subset = Subset(opencda_dataset, sampled_indices)
    
    data_loader = DataLoader(subset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # for test
    temp = subset[0]

    for i, batch_data in enumerate(data_loader):
        print(i)
        veh_frame_id = batch_data['ego']['veh_frame_id']
        batch_data['ego'].pop('veh_frame_id')
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)

        vis_save_path = os.path.join(output_path, '3d_veh_%s_%02d.png' % (veh_frame_id, time_delay))
        simple_vis.visualize(None,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='3d',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=False)
            
        vis_save_path = os.path.join(output_path, 'bev_veh_%s_%02d.png' % (veh_frame_id, time_delay))
        simple_vis.visualize(None,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=False)

        # vis_save_path = os.path.join(output_path, '3d_inf_%05d.png' % i)
        # simple_vis.visualize(None,
        #                     gt_box_tensor,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='bev',
        #                     vis_gt_box = vis_gt_box,
        #                     vis_pred_box = vis_pred_box,
        #                     left_hand=True)
        
        # vis_save_path = os.path.join(output_path, 'bev_inf_%05d.png' % i)
        # simple_vis.visualize(None,
        #                     gt_box_tensor,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='bev',
        #                     vis_gt_box = vis_gt_box,
        #                     vis_pred_box = vis_pred_box,
        #                     left_hand=True)
