# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict
import sys
sys.path.append(os.getcwd())

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis

from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange

# from opencood.models.fuse_modules.raindrop_attn_compensation import if_save_pt

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=80,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--pretrained_path', default='',
                        help='The path of the model need to be fine tuned.')
    parser.add_argument('--note', default="ir_thre_0_d_20", type=str, help='save folder name')
    parser.add_argument('--p', default=None, type=float, help='binomial probability')
    parser.add_argument('--two_stage', help='whether to use two stage training', default=0, type=int)
    opt = parser.parse_args()
    return opt


def main():
    # global if_save_pt
    # if_save_pt = False
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty'] 

    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # update binomial prob
    if 'binomial_p' in hypes and opt.p is not None:
        hypes['binomial_p'] = opt.p
    
    # This is used in visualization
    # left hand: OPV2V
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if "OPV2V" in hypes['test_dir'] else False



    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']
        
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model_diff(saved_path, model)
    
    if opt.pretrained_path: # load traj pred model
        saved_path = opt.pretrained_path
        pretrained_model_dict = torch.load(saved_path, map_location='cpu')
        diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in model.state_dict()}
        modified_pretrained_model_dict = OrderedDict()
        for k, v in pretrained_model_dict['model_dict'].items():
            modified_pretrained_model_dict.update({'matcher.compensate_motion.'+k: v})
        diff_keys = {k:v for k, v in modified_pretrained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        # diff_keys = {k:v for k, v in model.state_dict().items() if k not in pretrained_model_dict.keys()}
        # if diff_keys:
        #     print(f"!!! Created model has keys: {diff_keys.keys()}, \
        #         which are not in the model you have trained!!!")
        model.load_state_dict(modified_pretrained_model_dict, strict=False)
    
    model.eval()

    # setting noise
    np.random.seed(303)
    noise_setting = OrderedDict()
    noise_setting['add_noise'] = False
    
    # build dataset for each noise setting
    print('Dataset Building')
    print(f"No Noise Added.")
    hypes.update({"noise_setting": noise_setting})
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    noise_level = "no_noise"

    start_time = time.time()
    # infer_info = opt.fusion_method + f"{opt.cavnum}agent" + opt.note
    i = -1
    avg_time_delay = 0.0
    avg_sample_interval = 0.0
    for i, batch_data in tenumerate(data_loader):
        # if i < 90:
        #     continue # TODO: debug use

        # if i> 50:
        #     print("finished!")
            
        if batch_data is None:
            continue
        with torch.no_grad():
            # if i < 200:
            #     continue
            if opt.fusion_method == 'late':
                unit_time_delay = []
                unit_sample_interval = []
                for cav_id, cav_content in batch_data.items():
                    unit_time_delay.append(cav_content['debug']['time_diff'])
                    unit_sample_interval.append(cav_content['debug']['sample_interval'])
                avg_time_delay += (sum(unit_time_delay[1:])/len(unit_time_delay[1:]))
                avg_sample_interval += (float(sum(unit_sample_interval[1:]))/len(unit_sample_interval[1:]))
            if opt.fusion_method == 'intermediate':
                avg_time_delay += batch_data['ego']['avg_time_delay']
                avg_sample_interval += batch_data['ego']['avg_sample_interval']
            
            batch_data = train_utils.to_device(batch_data, device)
            uncertainty_tensor = None
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                if opt.two_stage == 1:
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion_flow_module(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
                    inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            else:
                raise NotImplementedError('Only no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego']['origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{opt.note}_%s'%(str(hypes['binomial_p'])))
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(pred_box_tensor,
                #                     gt_box_tensor,
                #                     batch_data['ego']['origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand,
                #                     uncertainty=uncertainty_tensor)
                
                try:
                    debug_path = batch_data['ego']['debug']['scene_name'] + '_' + batch_data['ego']['debug']['cav_id'] + '_' + batch_data['ego']['debug']['timestamp']
                    # print(debug_path)
                except:
                    debug_path = 'path'
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d_%s.png' % (i, debug_path))
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    uncertainty=uncertainty_tensor)
        torch.cuda.empty_cache()
    end_time = time.time()
    print("Time Consumed: %.2f minutes" % ((end_time - start_time)/60))
    
    avg_time_delay = (avg_time_delay/i) * 50 # unit is ms
    avg_sample_interval /= i
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, noise_level, avg_time_delay, avg_sample_interval, opt.note+'_'+str(hypes['binomial_p']))
    print("Module with sample interval expection: {}".format(hypes['binomial_n']*hypes['binomial_p']))


if __name__ == '__main__':
    main()
