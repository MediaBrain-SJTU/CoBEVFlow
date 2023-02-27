# -*- coding: utf-8 -*-
# Author: sizhewei @ 2023/1/27
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for Late fusion with past k frames on irregular OPV2V
TODO: train 部分因为暂时没有用到 可能存在bug 使用前需要检查
"""

from collections import OrderedDict
import os
import numpy as np
import torch
import math
import random
import copy
import sys
import time
import json
from scipy import stats
import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.keypoint_utils import bev_sample, get_keypoints
from opencood.data_utils.datasets import basedataset, intermediate_fusion_dataset_opv2v_irregular
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.utils.pose_utils import add_noise_data_dict, remove_z_axis
from opencood.utils.common_utils import read_json
from opencood.utils import box_utils
# from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

# from opencood.data_utils.datasets import build_dataset 

class LateFusionDatasetIrregular4Flow(intermediate_fusion_dataset_opv2v_irregular.IntermediateFusionDatasetIrregular):
    """
    This class is for late fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(LateFusionDatasetIrregular4Flow, self).__init__(params, visualize, train)

        print("=== Late fusion with non-ego cavs' past {} frames collected initialized! Expectation of sample interval is {}. ### {} ###  samples totally! ===".format(self.k, self.binomial_n * self.binomial_p, self.len_record[-1]))

    def __getitem__(self, idx):
        '''
        Returns:
        ------ 
        {
            # if train 
            # if test

        }
        '''
        # # debug use TODO:
        # if idx < 170 or idx > 270:
        #     return None
        base_data_dict = self.retrieve_base_data(idx)
        
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        # base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
        # during training, we return a random cav's data
        # only one vehicle is in processed_data_dict
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        ''' 
        Fetch useful info from base_data_dict, filter out too far cav.
        Return a dict match the point_pillar model.
        
        Params:
        ------
        base_data_dict : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            Structure: 
            {
                cav_id_1 : {
                    'ego' : true,
                    curr : {		(id)			#      |       | label
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string
                    },
                    past_k : {		# (k) totally
                        [0]:{		(id)			# pose | lidar | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string,
                            'time_diff': float,
                            'sample_interval': int
                        },
                        [1] : {},	(id-1)			# pose | lidar | label
                        ...,						# pose | lidar | label
                        [k-1] : {} (id-(k-1))		# pose | lidar | label
                    },
                    'debug' : {                     # debug use
                        scene_name : string         
                        cav_id : string
                    }
                    
                }, 
                cav_id_2 : {
                    'ego': false, 
                    curr : 	{		(id)			#      |       | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string
                    },
                    past_k: {		# (k) totally
                        [0] : {		(id - \tau)		# pose | lidar | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string,
                            'time_diff': float,
                            'sample_interval': int
                        }			
                        ..., 						# pose | lidar | label
                        [k-1]:{}  (id-\tau-(k-1))	# pose | lidar | label
                    },
                }, 
                ...
            }

        Returns:
        ------ 
        {
            'ego' : {
                'if_no_point' : bool,
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego'),
                    'timestamp' : string ('past_k'-0-timestamp)
                },
                'origin_lidar' : (if visualize) original lidar
                'curr_feature' : current processed feature at cav current view
                'object_bbx_center' : 
                'object_ids' :
                'label_dict' : 
                'trans_mat_cavpast0_2_egocurr' 
                'past_k_processed_lidar' : list, len is k
                'past_k_time_diff' :  list, len is k
                'past_k_trans_mat_cavpast_2_cavpast0' : np.array, [k, 4, 4]
                'transformation_matrix_curr': cavcurr to egocurr, [4, 4]
                'past0_pose' 
            },
            cav_id: { ... } 
        }
        '''
        processed_data_dict = OrderedDict()

        # first find the ego vehicle's lidar pose
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['curr']['params']['lidar_pose']
                break	
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # scene_name = base_data_dict[ego_id]['debug']['scene_name']
        
        too_far = []
        cav_id_list = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            # for non-ego cav, we use the latest frame's pose
            distance = \
                math.sqrt( \
                    (selected_cav_base['past_k'][0]['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + \
                    (selected_cav_base['past_k'][0]['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue
            cav_id_list.append(cav_id)  
        # filter those out of communicate range
        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        if self.visualize:
            projected_lidar_stack = []

        illegal_cav = []
        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]

            selected_cav_processed = self.get_item_single_car_test(
                selected_cav_base,
                ego_lidar_pose
            )
            '''
            selected_cav_processed : {
                'if_no_point' : False / True ,
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego')
                },
                'trans_mat_cavpast0_2_egocurr' : ,
                'past_k' : {
                    [0] : {
                        'origin_lidar' : 
                        'processed_lidar' : 
                        'anchor_box' : 
                        'object_bbx_center' : 
                        'object_bbx_mask' : 
                        'object_ids' : 
                        'label_dict' : 
                        'trans_mat_cavpast_2_cavpast0' : 
                    }, 
                    [1] : { ... },
                    ...,
                    [k-1] : { ... }
                }, 
                'curr' : {}
            }
            selected_cav_processed: dict{
                'if_no_point' : bool,
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego'),
                    'timestamp' : string ('past_k'-0-timestamp)
                },
                'origin_lidar' : (if visualize) original lidar
                'curr_feature' : current processed feature at cav current view
                'object_bbx_center' : 
                'object_ids' :
                'anchor_box' :   
                'label_dict' : 
                'trans_mat_cavpast0_2_egocurr' 
                'past_k_processed_lidar' : list, len is k
                'past_k_time_diff' :  list, len is k
                'past_k_trans_mat_cavpast_2_cavpast0' : np.array, k, 4, 4 
                'past0_pose'
            }
            '''

            if selected_cav_processed['if_no_point']: # 把点的数量不合法的车排除
                illegal_cav.append(cav_id)
                # # 把出现不合法sample的 场景、车辆、时刻 记录下来:
                # illegal_path = os.path.join(base_data_dict[cav_id]['debug']['scene'], cav_id, base_data_dict[cav_id]['past_k'][0]['timestamp']+'.npy')
                # illegal_path_list.add(illegal_path)
                # print(illegal_path)
                continue
            
            # cav_lidar_pose_past = selected_cav_base['past_k'][0]['params']['lidar_pose']
            # transformation_matrix_past = x1_to_x2(cav_lidar_pose_past, ego_lidar_pose)
            # selected_cav_processed.update({'transformation_matrix_past': transformation_matrix_past})

            cav_lidar_pose_curr = selected_cav_base['curr']['params']['lidar_pose']
            transformation_matrix_curr = x1_to_x2(cav_lidar_pose_curr, ego_lidar_pose)
            selected_cav_processed.update({'transformation_matrix_curr': transformation_matrix_curr})
            
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})
        
        # filter out cav with no point:
        for cav_id in illegal_cav:
            base_data_dict.pop(cav_id)

        return processed_data_dict

    def get_item_single_car_test(self, selected_cav_base, ego_lidar_pose):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            Structure : {
                'ego' : true,
                curr : {		(id)			#      |       | label
                    'params': (yaml),
                    'lidar_np': (numpy),
                    'timestamp': string
                },
                past_k : {		# (k) totally
                    [0]:{		(id)			# pose | lidar | label
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)			# pose | lidar | label
                    ...,						# pose | lidar | label
                    [k-1] : {} (id-(k-1))		# pose | lidar | label
                },
                'debug' : {                     # debug use
                    scene_name : string         
                    cav_id : string
                }    
            }

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            Structure : {
                'if_no_point' : False / True ,
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego'),
                    'timestamp' : string ('past_k'-0-timestamp)
                },
                'trans_mat_cavpast0_2_egocurr' : 
                'past_k' : {
                    [0] : {
                        'origin_lidar' : 
                        'processed_lidar' : 
                        'anchor_box' : 
                        'object_bbx_center' : 
                        'object_bbx_mask' : 
                        'object_ids' : 
                        'label_dict' : 
                        'time_diff' : 
                        'trans_mat_cavpast_2_cavpast0' : 
                    }
                    [1] : { ... },
                    ...
                    [k-1] : { ... }
                },
                'curr' : {
                    'origin_lidar' : 
                    'processed_lidar' : 
                    'anchor_box' : 
                    'object_bbx_center' : 
                    'object_bbx_mask' : 
                    'object_ids' : 
                    'label_dict' : 
                }
            }
        selected_cav_processed: dict{
            'if_no_point' : bool,
            'debug' : {
                'scene_name' : string,
                'cav_id' : string,
                'time_diff': float (0.0 if 'ego'),
                'sample_interval': int (0 if 'ego'),
                'timestamp' : string ('past_k'-0-timestamp)
            },
            'origin_lidar' : (if visualize) original lidar
            'curr_feature' : current processed feature at cav current view
            'object_bbx_center' : 
            'object_bbx_mask' :
            'object_ids' :
            'label_dict' : 
            'anchor_box' : 
            'trans_mat_cavpast0_2_egocurr'                  cav-past0 to ego-curr
            'trans_mat_cavpast0_2_cavcurr'                  cav-past0 to cav-curr
            'past_k_processed_lidar' : list, len is k
            'past_k_time_diff' :  list, len is k
            'past_k_trans_mat_cavpast_2_cavpast0' : np.array, k, 4, 4 .  cav-past to cav-past0
            'past0_pose'
        }
        """
        selected_cav_processed = {}
        ego_lidar_pose = ego_lidar_pose

        if_no_point = False
        
        ###################### for current frame ######################
        processed_part = {}
        processing_base = selected_cav_base['curr']

        # filter lidar
        lidar_np = processing_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)
        
        # tag illegal situation
        if lidar_np.shape[0] == 0: # 没有点留下
            selected_cav_processed.update({'if_no_point': True})
            return selected_cav_processed

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([processing_base], processing_base['params']['lidar_pose'])  

        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask) # TODO: check

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        # processed_part.update({'processed_lidar': lidar_dict})
        selected_cav_processed.update({'curr_feature': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        processed_part.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'object_ids': object_ids, 
            'anchor_box': anchor_box})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        
        selected_cav_processed.update({'label_dict': label_dict})

        # selected_cav_processed.update({'curr' : processed_part})

        # trans mat : past to past 0 / past 0 to curr
        trans_mat_cavpast0_2_egocurr = x1_to_x2(selected_cav_base['past_k'][0]['params']['lidar_pose'], ego_lidar_pose)
        selected_cav_processed.update({'trans_mat_cavpast0_2_egocurr': trans_mat_cavpast0_2_egocurr})

        trans_mat_cavpast0_2_cavcurr = x1_to_x2(selected_cav_base['past_k'][0]['params']['lidar_pose'], \
             selected_cav_base['curr']['params']['lidar_pose'])
        selected_cav_processed.update({'trans_mat_cavpast0_2_cavcurr': trans_mat_cavpast0_2_cavcurr})

        ################### for past k frames ########################
        """
        'processed_lidar' : 
        'time_diff' : 
        'trans_mat_cavpast_2_cavpast0' :  
        """
        # if self.visualize:
        #     past_k_origin_lidar = []
        past_k_processed_lidar = []
        # past_k_anchor_box = []
        # past_k_object_bbx_center = []
        # past_k_object_bbx_mask = [] 
        # past_k_object_ids = []
        # past_k_label_dict = []
        past_k_time_diff = [] 
        past_k_trans_mat_cavpast_2_cavpast0 = [] 

        # selected_cav_processed['past_k'] = OrderedDict()
        for i in range(self.k):
            # processed_part = {}
            processing_base = selected_cav_base['past_k'][i]
            
            # trans mat : past to past 0 / past 0 to curr
            trans_mat_cavpast_2_cavpast0 = x1_to_x2(processing_base['params']['lidar_pose'], selected_cav_base['past_k'][0]['params']['lidar_pose'])
            # processed_part.update({'trans_mat_cavpast_2_cavpast0', trans_mat_cavpast_2_cavpast0})
            past_k_trans_mat_cavpast_2_cavpast0.append(trans_mat_cavpast_2_cavpast0)

            # filter lidar
            lidar_np = processing_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(lidar_np,
                                            self.params['preprocess'][
                                                'cav_lidar_range'])
            # remove points that hit ego vehicle
            lidar_np = mask_ego_points(lidar_np)
            
            # tag illegal situation
            if lidar_np.shape[0] == 0: # 没有点留下
                selected_cav_processed.update({'if_no_point': True})
                return selected_cav_processed

            # generate the bounding box(n, 7) under the cav's space
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([processing_base], processing_base['params']['lidar_pose'])  

            # data augmentation
            lidar_np, object_bbx_center, object_bbx_mask = \
                self.augment(lidar_np, object_bbx_center, object_bbx_mask) # TODO: check

            # if self.visualize:
            #     # processed_part.update({'origin_lidar': lidar_np})
            #     past_k_origin_lidar.append(lidar_np)

            # pre-process the lidar to voxel/bev/downsampled lidar
            lidar_dict = self.pre_processor.preprocess(lidar_np)
            # processed_part.update({'processed_lidar': lidar_dict})
            past_k_processed_lidar.append(lidar_dict)

            # generate the anchor boxes
            # anchor_box = self.post_processor.generate_anchor_box()
            # processed_part.update({'anchor_box': anchor_box})
            # past_k_anchor_box.append(anchor_box)

            # processed_part.update({'object_bbx_center': object_bbx_center,
            #                             'object_bbx_mask': object_bbx_mask,
            #                             'object_ids': object_ids})
            
            # processed_part.update({'time_diff': selected_cav_base['past_k'][i]['time_diff']})
            past_k_time_diff.append(selected_cav_base['past_k'][i]['time_diff'])

            # generate targets label
            # label_dict = \
            #     self.post_processor.generate_label(
            #         gt_box_center=object_bbx_center,
            #         anchors=anchor_box,
            #         mask=object_bbx_mask)
            
            # processed_part.update({'label_dict': label_dict})

            # selected_cav_processed['past_k'][i] = processed_part

        # past_k_time_diff = np.array(past_k_time_diff)  # len = k
        merged_past_k_processed_lidar = self.merge_cav_past_k_features_to_dict(past_k_processed_lidar)
        past_k_trans_mat_cavpast_2_cavpast0 = np.stack(past_k_trans_mat_cavpast_2_cavpast0, axis=0)  # k, 4, 4

        selected_cav_processed.update({
            'past_k_processed_lidar': merged_past_k_processed_lidar, 
            'past_k_time_diff': past_k_time_diff,
            'past_k_trans_mat_cavpast_2_cavpast0': past_k_trans_mat_cavpast_2_cavpast0})

        selected_cav_processed.update({
            'past0_pose' : selected_cav_base['past_k'][0]['params']['lidar_pose']
        })
        
        ############################################################################
        debug_part = {}
        # print(selected_cav_base['debug'])
        debug_part.update({'time_diff': selected_cav_base['past_k'][0]['time_diff'], 
                            'sample_interval': selected_cav_base['past_k'][0]['sample_interval'],
                            'scene_name': selected_cav_base['debug']['scene'],
                            'cav_id': selected_cav_base['debug']['cav_id'],
                            'timestamp': selected_cav_base['past_k'][0]['timestamp']})
        selected_cav_processed.update({'if_no_point': if_no_point,
                                        'debug': debug_part})

        return selected_cav_processed
    '''
    def get_item_single_car(self, selected_cav_base, idx=-1):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information, 
            structure: {
                'ego' : true / false,
                curr : {		(id)							#      |       | label
                    'params': (yaml),
                    'lidar_np': (numpy),
                    'timestamp': string
                },
                past_k : {		# (k) totally
                    [0]:{		(id) / (id-\tau-0)				# pose | lidar | label
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string
                    },
                    [1] : {},	(id-1) / (id-\tau-1))			# pose | lidar | label
                    ...,										# pose | lidar | label
                    [k-1] : {} 	(id-(k-1)) / (id-\tau-(k-1))	# pose | lidar | label
                }	
            }, 
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        idx: int,
            debug use.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            {
                single_label_dict:		# single view label
                curr_feature:           # curr_feature,
                object_bbx_center:		# ego view label. np.ndarray. Shape is (max_num, 7)
                object_ids:				# ego view label index. list. length is (max_num, 7)
                'curr_pose':			# current pose, list, len = 6
                'past_k_poses': 		# list of past k frames' poses
                'past_k_features': 		# list of past k frames' lidar
                'past_k_time_diffs': 	# list of past k frames' time diff with current frame
                # 'past_k_tr_mats': 		# list of past k frames' transformation matrix to current ego coordinate
                'past_k_label_dicts'    # [], [TBD] list of past k frames' label dict 
            }
        """
        selected_cav_processed = {}

        # curr lidar feature
        lidar_np = selected_cav_base['curr']['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np) # remove points that hit itself

        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        curr_feature = self.pre_processor.preprocess(lidar_np)
        
        # # past k transfomation matrix
        # past_k_tr_mats = []
        # past k lidars
        past_k_features = []
        # past k poses
        past_k_poses = []
        # past k timestamps
        past_k_time_diffs = []
        # past k sample invervals
        past_k_sample_interval = []
        
        # past k label 
        past_k_label_dicts = [] # todo 这个部分可以删掉

        # 判断点的数量是否合法
        if_no_point = False

        # past k frames [trans matrix], [lidar feature], [pose], [time interval]
        for i in range(self.k):
            # # 1. trans matrix
            # transformation_matrix = \
            #     x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray
            # past_k_tr_mats.append(transformation_matrix)

            # 2. lidar feature
            lidar_np = selected_cav_base['past_k'][i]['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np) # remove points that hit itself
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            processed_features = self.pre_processor.preprocess(lidar_np)
            past_k_features.append(processed_features)

            if lidar_np.shape[0] == 0: # 没有点留下
                if_no_point = True
            
            # 3. pose
            past_k_poses.append(selected_cav_base['past_k'][i]['params']['lidar_pose'])

            # 4. time interval and sample interval
            past_k_time_diffs.append(selected_cav_base['past_k'][i]['time_diff'])
            past_k_sample_interval.append(selected_cav_base['past_k'][i]['sample_interval'])

        # past_k_tr_mats = np.stack(past_k_tr_mats, axis=0) # (k, 4, 4)

        # curr label at single view
        # opencood/data_utils/post_processor/base_postprocessor.py
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([selected_cav_base['curr']], selected_cav_base['curr']['params']['lidar_pose'])  
        # generate the anchor boxes
        # opencood/data_utils/post_processor/voxel_postprocessor.py
        anchor_box = self.anchor_box
        label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
            )
        selected_cav_processed.update({})
        
        # # curr label at ego view
        # object_bbx_center, object_bbx_mask, object_ids = \
        #     self.generate_object_center([selected_cav_base['curr']], ego_pose)
            
        selected_cav_processed.update(
            {"single_label_dict": label_dict,
             "curr_feature": curr_feature,
             'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'curr_pose': selected_cav_base['curr']['params']['lidar_pose'],
            #  'past_k_tr_mats': past_k_tr_mats,
             'past_k_poses': past_k_poses,
             'past_k_features': past_k_features,
             'past_k_time_diffs': past_k_time_diffs,
             'past_k_sample_interval': past_k_sample_interval,
             'past_k_label_dicts': past_k_label_dicts,
             'if_no_point': if_no_point
             })

        return selected_cav_processed
    '''
    def merge_cav_past_k_features_to_dict(self, processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for time_id in range(self.k):
            for feature_name, feature in processed_feature_list[time_id].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict

    def collate_batch_train(self, batch):
        '''
        Parameters:
        ----------
        batch[i]['ego'] structure:
        {   
            'single_object_dict_stack': single_label_dict_stack,
            'curr_processed_lidar': merged_curr_feature_dict,
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': mask,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'anchor_box': anchor_box,
            'processed_lidar': merged_feature_dict,
            'label_dict': label_dict,
            'cav_num': cav_num,
            'pairwise_t_matrix': pairwise_t_matrix,
            'curr_lidar_poses': curr_lidar_poses,
            'past_k_lidar_poses': past_k_lidar_poses,
            'sample_idx': idx,
            'cav_id_list': cav_id_list,
            'past_k_time_diffs': past_k_time_diffs_stack, 
                list of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
        }
        '''
        for i in range(len(batch)):
            if batch[i] is None:
                return None
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        single_object_label = []

        pos_equal_one_single = []
        neg_equal_one_single = []
        targets_single = []

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        curr_processed_lidar_list = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        curr_lidar_pose_list = []
        past_k_lidar_pose_list = []
        past_k_label_list = []
        # store the time interval of each feature map
        past_k_time_interval = []
        past_k_sample_inverval = []
        past_k_avg_sample_interval = []
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # for debug use:
        time_consume = np.zeros_like(batch[0]['ego']['times'])

        if self.visualize:
            origin_lidar = []
        
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            single_object_label.append(ego_dict['single_object_dict_stack'])

            pos_equal_one_single.append(ego_dict['single_object_dict_stack']['pos_equal_one'])
            neg_equal_one_single.append(ego_dict['single_object_dict_stack']['neg_equal_one'])
            targets_single.append(ego_dict['single_object_dict_stack']['targets'])

            curr_processed_lidar_list.append(ego_dict['curr_processed_lidar'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            curr_lidar_pose_list.append(ego_dict['curr_lidar_poses']) # ego_dict['curr_lidar_pose'] is np.ndarray [N,6]
            past_k_lidar_pose_list.append(ego_dict['past_k_lidar_poses']) # ego_dict['past_k_lidar_pose'] is np.ndarray [N,k,6]
            past_k_time_interval.append(ego_dict['past_k_time_diffs']) # ego_dict['past_k_time_diffs'] is np.array(), len=nxk
            past_k_sample_inverval.append(ego_dict['past_k_sample_interval']) # ego_dict['past_k_sample_interval'] is np.array(), len=nxk
            past_k_avg_sample_interval.append(ego_dict['avg_sample_interval']) # ego_dict['avg_sample_interval'] is float
            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            # pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            # past_k_label_list.append(ego_dict['past_k_label_dicts'])

            time_consume += ego_dict['times']
            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        
        
        # single_object_label = self.post_processor.collate_batch(single_object_label)
        single_object_label = { "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                                "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                                 "targets": torch.cat(targets_single, dim=0)}
                                 
        # collate past k single view label from different batch [B, cav_num, k, 100, 252, 2]...
        past_k_single_label_torch_dict = self.post_processor.collate_batch(past_k_label_list)
        
        # collate past k time interval from different batch, (B, )
        past_k_time_interval = np.hstack(past_k_time_interval)
        past_k_time_interval = torch.from_numpy(past_k_time_interval)

        # collate past k sample interval from different batch, (B, )
        past_k_sample_inverval = np.hstack(past_k_sample_inverval)
        past_k_sample_inverval = torch.from_numpy(past_k_sample_inverval)

        past_k_avg_sample_interval = np.array(past_k_avg_sample_interval)
        avg_sample_interval = sum(past_k_avg_sample_interval) / len(past_k_avg_sample_interval)

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        # （B, max_num)
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        
        curr_merged_feature_dict = self.merge_features_to_dict(curr_processed_lidar_list)
        curr_processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(curr_merged_feature_dict)
        # processed_lidar_list: list, len is 6. [batch_i] is OrderedDict, 3 keys: {'voxel_features': , ...}
        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        curr_lidar_pose = torch.from_numpy(np.concatenate(curr_lidar_pose_list, axis=0))
        # [[N1, k, 6], [N2, k, 6]...] -> [(N1+N2+...), k, 6]
        past_k_lidar_pose = torch.from_numpy(np.concatenate(past_k_lidar_pose_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav, k, 4, 4)
        # pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        # label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        # for debug use: 
        time_consume = torch.from_numpy(time_consume)

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'single_object_label': single_object_label,
                                   'curr_processed_lidar': curr_processed_lidar_torch_dict,
                                   'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'single_past_dict': past_k_single_label_torch_dict,
                                   'object_ids': object_ids[0],
                                #    'pairwise_t_matrix': pairwise_t_matrix,
                                   'curr_lidar_pose': curr_lidar_pose,
                                   'past_lidar_pose': past_k_lidar_pose,
                                   'past_k_time_interval': past_k_time_interval,
                                   'past_k_sample_inverval': past_k_sample_inverval,
                                   'avg_sample_interval': avg_sample_interval,
                                   'times': time_consume})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})
            
        # if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and \
        #     (output_dict['ego']['processed_lidar']['voxel_coords'][:, 0].max().int().item() + 1) != record_len.sum().int().item():
        #     return None

        return output_dict

    def collate_batch_test(self, batch):
        """
        Parameters:
        -----------
        batch: list, len = batch_size
        [0] : {
            'ego' / cav_id : {
                'if_no_point' : bool,
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego'),
                    'timestamp' : string ('past_k'-0-timestamp)
                },
                'origin_lidar' : (if visualize) original lidar
                'curr_feature' : current processed feature at cav current view
                'object_bbx_center' : 
                'object_ids' :
                'label_dict' : 
                'trans_mat_cavpast0_2_egocurr':  4, 4
                'trans_mat_cavpast0_2_cavcurr':  4, 4
                'past_k_processed_lidar' : list, len is k
                'past_k_time_diff' :  list, len is k
                'past_k_trans_mat_cavpast_2_cavpast0' : np.array, [k, 4, 4]
                'past0_pose'
                'transformation_matrix_curr': cavcurr to egocurr, [4, 4]
            },     
        } 

        Returns:
        ------
        Structure: {
            'ego' / cav_id : {
                'anchor_box' : ,
                'object_bbx_center':                    curr,
                'object_bbx_mask':                      curr,
                'object_ids':                           curr,
                'label_dict':                           curr,
                'processed_lidar':                      past_k 0 ,
                # 'transformation_matrix':                cav-past to ego-curr,
                'transformation_matrix_clean':          cav-curr to ego-curr, for viz
                'trans_mat_past_2_past0_torch' :    cav-past to cav-past0
                'trans_mat_cavpast0_2_cavcurr_torch':   cav-past0 to cav-curr
                'trans_mat_past0_2_curr_torch' :    cav-past0 to ego-curr
                'debug' : {
                    'scene_name' : string,
                    'cav_id' : string,
                    'time_diff': float (0.0 if 'ego'),
                    'sample_interval': int (0 if 'ego')
                },
                'origin_lidar' :                        cav-curr in ego view
            },
            cav_id : { ... }
        }
        """
        if batch[0] is None:
            return None
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            output_dict[cav_id].update({'past_k_time_diff': 
                torch.from_numpy(np.array(
                    cav_content['past_k_time_diff']))})
            
            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content['anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix_curr']
                origin_lidar = [cav_content['origin_lidar']] # TODO: check

                if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                    # print(cav_id)
                    import copy
                    projected_lidar = copy.deepcopy(cav_content['origin_lidar']) # TODO: check
                    projected_lidar[:, :3] = \
                        box_utils.project_points_by_matrix_torch(
                            projected_lidar[:, :3],
                            transformation_matrix)
                    projected_lidar_list.append(projected_lidar)

            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']]) # TODO: check
            
            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    cav_content['past_k_processed_lidar'])
            

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation_matrix_torch = \
            #     torch.from_numpy(
            #         np.array(cav_content['transformation_matrix_past'])).float()
            
            trans_mat_past_2_past0_torch = \
                torch.from_numpy(
                    np.array(cav_content['past_k_trans_mat_cavpast_2_cavpast0'])
                ).float()

            trans_mat_cavpast0_2_cavcurr_torch = \
                torch.from_numpy(
                    np.array(cav_content['trans_mat_cavpast0_2_cavcurr'])
                ).float()

            trans_mat_cavpast0_2_eogcurr_torch = \
                torch.from_numpy(
                    np.array(cav_content['trans_mat_cavpast0_2_egocurr'])
                ).float()
            
            # late fusion training, no noise
            transformation_matrix_clean_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix_curr'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'object_ids': object_ids,
                                        'label_dict': label_torch_dict,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        # 'transformation_matrix': transformation_matrix_torch,
                                        'transformation_matrix_clean': transformation_matrix_clean_torch,
                                        'trans_mat_past_2_past0_torch':trans_mat_past_2_past0_torch,
                                        'trans_mat_cavpast0_2_cavcurr_torch': trans_mat_cavpast0_2_cavcurr_torch,
                                        'trans_mat_past0_2_curr_torch':trans_mat_cavpast0_2_eogcurr_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

            output_dict[cav_id].update({'debug' : cav_content['debug']})

        pairwise_t_matrix = torch.from_numpy(self.get_pairwise_transformation_w_time_diff(batch, self.max_cav))
        output_dict['ego'].update({'pairwise_t_matrix': pairwise_t_matrix})
        
        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
            # output_dict['ego'].update({'projected_lidar_list': projected_lidar_list})

        return output_dict

    def post_process_no_fusion(self, data_dict, output_dict_ego, return_uncertainty=False):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego)
            return pred_box_tensor, pred_score, gt_box_tensor
            
    def generate_pred_bbx_frames(self, data_dict, output_dict):
        box_results = self.post_processor.flow_post_process(data_dict, output_dict, self.k)
        return box_results

    def generate_gt_bbx_cav_curr(self, data_dict):
        return self.post_processor.generate_gt_bbx_cav_curr(data_dict)

    def post_process_updated(self, data_dict, output_dict):
        return self.post_processor.post_process_fuse_updated_frame(data_dict, output_dict)


    def post_process_for_intermediate(self, data_dict, output_dict):
        pred_box_tensor, pred_score = \
            self.post_processor.post_process_for_intermediate(data_dict, output_dict)

        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
    
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)

        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation_w_time_diff(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents
        Note, this function caculate the trans_mat between cav-past-0 and ego-curr

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['past0_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['curr']['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
    
    def get_past_k_pairwise_transformation2ego(self, base_data_dict, ego_pose, max_cav):
        """
        Get transformation matrixes accross different agents to curr ego at all past timestamps.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.
        
        ego_pose : list
            ego pose

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The transformation matrix each cav to curr ego at past k frames.
            shape: (L, k, 4, 4), L is the max cav number in a scene, k is the num of past frames
            pairwise_t_matrix[i, j] is T i_to_ego at past_j frame
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, self.k, 1, 1)) # (L, k, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                past_k_poses = []
                for time_id in range(self.k):
                    past_k_poses.append(x_to_world(cav_content['past_k'][time_id]['params']['lidar_pose']))
                t_list.append(past_k_poses) # Twx
            
            ego_pose = x_to_world(ego_pose)
            for i in range(len(t_list)): # different cav
                if i!=0 :
                    for j in range(len(t_list[i])): # different time
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

if __name__ == '__main__':   
    
    def train_parser():
        parser = argparse.ArgumentParser(description="synthetic data generation")
        parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                            help='data generation yaml file needed ')
        parser.add_argument('--model_dir', default='',
                            help='Continued training path')
        parser.add_argument('--fusion_method', '-f', default="intermediate",
                            help='passed to inference.')
        opt = parser.parse_args()
        return opt
    
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('### Dataset Building ... ###')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)