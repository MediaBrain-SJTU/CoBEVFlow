# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified: sizhewei @ 2022/11/05

"""
Dataset class for intermediate fusion with past k frames
"""

from collections import OrderedDict
import os
import numpy as np
import torch
import math
import copy

import time
import json
import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.keypoint_utils import bev_sample, get_keypoints
from opencood.data_utils.datasets import basedataset
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


class IntermediateFusionDatasetMultisweep(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):

        self.times = []

        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'num_sweep_frames' in params:    # number of frames we use in LSTM
            self.k = params['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in params:          # number of time delay
            self.tau = params['time_delay'] 
        else:
            self.tau = 0

        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders_name = sorted([x
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        '''
        scenario_database Structure: 
        {
            scenario_id : {
                cav_1 : {
                    'ego' : true / false , 
                    timestamp1 : {
                        yaml: path,
                        lidar: path, 
                        cameras: list of path
                    },
                    ...
                },
                ...
            }
        }
        '''
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x 
                               for x in os.listdir(scenario_folder) if 
                               os.path.isdir(os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = self.extract_timestamps(yaml_files)	

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    # camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    # self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        # camera_files

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:  # ego 
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    num_ego_timestamps = len(timestamps) - (self.tau + self.k - 1)		# 从第 tau+k 个往后, store 0 时刻的 time stamp
                    if not self.len_record:
                        self.len_record.append(num_ego_timestamps)
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + num_ego_timestamps)

                else:
                    self.scenario_database[i][cav_id]['ego'] = False
                    

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        # TODO: check pre_process & post_processor 是否符合新的数据结构 
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        self.anchor_box = self.post_processor.generate_anchor_box()

        print("OPV2V Multi-sweep dataset with non-ego cavs' {} time delay and past {} frames collected initialized! \
                {} samples totally!".format(self.tau, self.k, self.len_record[-1]))

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            Structure: 
            {
                cav_id_1 : {
                    'ego' : true,
                    curr : {		(id)			# pose | lidar | label
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string
                    },
                    past_k : {		# (k) totally
                        [0]:{		(id)			# pose | lidar | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string
                        },
                        [1] : {},	(id-1)			# pose | lidar | label
                        ...,						# pose | lidar | label
                        [k-1] : {} (id-(k-1))		# pose | lidar | label
                    }
                    
                }, 
                cav_id_2 : {
                    'ego': false, 
                    curr : 	{		(id)			# pose | lidar | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string
                    },
                    past_k: {		# (k) totally
                        [0] : {		(id - \tau)		# pose | lidar | label
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string
                        }			
                        ..., 						# pose | lidar | label
                        [k-1]:{}  (id-\tau-(k-1))	# pose | lidar | label
                    },
                }, 
                ...
            }
        """
        # we loop the accumulated length list to get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        debug_times = np.zeros((3,))
        start_time = time.time()

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            '''
            cav_content 
            {
                'ego' : true / false , 
                timestamp1 : {
                    yaml: path,
                    lidar: path, 
                    cameras: list of path
                },
                ...
            },
            '''
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            data[cav_id]['past_k'] = OrderedDict()
            
            # current frame, for co-perception lable use
            data[cav_id]['curr'] = {}
            timestamp_index = idx if scenario_index == 0 else \
                        idx - self.len_record[scenario_index - 1]
            timestamp_index = timestamp_index + self.tau + self.k - 1
            timestamp_key = list(cav_content.items())[timestamp_index][0]
            
            time_s = time.time()
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['curr']['params'] = json.load(f)
            else:
                data[cav_id]['curr']['params'] = \
                        load_yaml(cav_content[timestamp_key]['yaml'])
            time_e = time.time()
            debug_times[0] += (time_e - time_s)

            time_s = time.time()
            # load lidar file: npy is faster than pcd
            npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
            if os.path.exists(npy_file):
                data[cav_id]['curr']['lidar_np'] = np.load(npy_file)
            else:
                data[cav_id]['curr']['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])
            time_e = time.time()
            debug_times[1] += (time_e - time_s)

            data[cav_id]['curr']['timestamp'] = \
                    timestamp_key
            data[cav_id]['curr']['time_diff'] = 0

            # past k frames, pose | lidar | label(for single view confidence map generator use)
            if data[cav_id]['ego']:
                temp = self.tau
            else:
                temp = 0
            for i in range(self.k):
                # check the timestamp index
                data[cav_id]['past_k'][i] = OrderedDict()
                timestamp_index = idx if scenario_index == 0 else \
                    idx - self.len_record[scenario_index - 1] # TODO: 这里面原来错加了一个i，记得改正
                timestamp_index = timestamp_index + self.k - 1 - i + temp
                timestamp_key = list(cav_content.items())[timestamp_index][0]
                # load the corresponding data into the dictionary
                time_s = time.time()
                # load param file: json is faster than yaml
                json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data[cav_id]['past_k'][i]['params'] = json.load(f)
                else:
                    data[cav_id]['past_k'][i]['params'] = \
                        load_yaml(cav_content[timestamp_key]['yaml'])
                time_e = time.time()
                debug_times[0] += (time_e - time_s)

                time_s = time.time()
                # load lidar file: npy is faster than pcd
                npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
                if os.path.exists(npy_file):
                    data[cav_id]['past_k'][i]['lidar_np'] = np.load(npy_file)
                else:
                    data[cav_id]['past_k'][i]['lidar_np'] = \
                            pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])
                time_e = time.time()
                debug_times[1] += (time_e - time_s)

                data[cav_id]['past_k'][i]['timestamp'] = \
                    timestamp_key
                data[cav_id]['past_k'][i]['time_diff'] = \
                    self.dist_time(timestamp_key, data[cav_id]['curr']['timestamp'])

        end_time = time.time()

        debug_times[2] += (end_time - start_time)

        return data

    def __getitem__(self, idx):
        '''
        Returns:
        ------ 
        {
            'single_object_dict_stack': single_label_dict_stack,
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
                np.array of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
        }
        '''
        # base_data_dict
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        # first find the ego vehicle's lidar pose
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['curr']['params']['lidar_pose']
                # print(ego_lidar_pose)
                # ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break	
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        too_far = []
        curr_lidar_pose_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

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

            curr_lidar_pose_list.append(selected_cav_base['curr']['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)  

        single_label_dict_stack = []
        object_stack = []
        object_id_stack = []
        curr_pose_stack = []
        curr_feature_stack = []
        past_k_pose_stack = []
        past_k_features_stack = [] 
        past_k_time_diffs_stack = []
        past_k_tr_mats = []
        past_k_label_dicts_stack = []

        if self.visualize:
            projected_lidar_stack = []
        
        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            '''
            selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            {
                'single_label_dict':	# single view label
                'object_bbx_center':	# ego view label. np.ndarray. Shape is (max_num, 7)
                'object_ids':			# ego view label index. list. length is (max_num, 7)
                'curr_pose':			# current pose, list, len = 6
                'past_k_poses': 		# list of past k frames' poses
                'past_k_features': 		# list of past k frames' lidar
                'past_k_time_diffs': 	# list of past k frames' time diff with current frame
                'past_k_tr_mats': 		# list of past k frames' transformation matrix to current ego coordinate
                'past_k_label_dicts'    # [], [TBD] list of past k frames' label dict 
            }
            '''
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose, 
                idx
            )

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
            
            # single view feature
            curr_feature_stack.append(selected_cav_processed['curr_feature'])
            # single view label
            single_label_dict_stack.append(selected_cav_processed['single_label_dict'])

            # curr ego view label
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

            # current pose: N, 6
            curr_pose_stack.append(selected_cav_processed['curr_pose']) 
            # features: N, k, 
            past_k_features_stack.append(selected_cav_processed['past_k_features'])
            # poses: N, k, 6
            past_k_pose_stack.append(selected_cav_processed['past_k_poses'])
            # time differences: N, k
            past_k_time_diffs_stack += selected_cav_processed['past_k_time_diffs']
            # past k frames to ego pose trans matrix, list of len=N, past_k_tr_mats[i]: ndarray(k, 4, 4)
            past_k_tr_mats.append(selected_cav_processed['past_k_tr_mats'])
            # past k label dict: N, k, object_num, 7
            # past_k_label_dicts_stack.append(selected_cav_processed['past_k_label_dicts'])
        # {pos: array[num_cav, k, 100, 252, 2], neg: array[num_cav, k, 100, 252, 2], target: array[num_cav, k, 100, 252, 2]}
        # past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts_stack)

        merged_curr_feature_dict = self.merge_features_to_dict(curr_feature_stack)
        
        single_label_dict = self.post_processor.collate_batch(single_label_dict_stack)

        past_k_time_diffs_stack = np.array(past_k_time_diffs_stack)

        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)
        
        pairwise_t_matrix = \
            self.get_past_k_pairwise_transformation2ego(base_data_dict, 
            ego_lidar_pose, self.max_cav) # np.tile(np.eye(4), (max_cav, self.k, 1, 1)) (L, k, 4, 4)

        curr_lidar_poses = np.array(curr_pose_stack).reshape(-1, 6)  # (N_cav, 6)
        past_k_lidar_poses = np.array(past_k_pose_stack).reshape(-1, self.k, 6)  # (N, k, 6)

        # exclude all repetitive objects    
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(cav_id_list)
        # past_k_features_stack: list, len is num_cav. [i] is list, len is k. [cav_id][time_id] is Orderdict, {'voxel_features': array, ...}
        merged_feature_dict = self.merge_past_k_features_to_dict(past_k_features_stack)

        # generate the anchor boxes
        anchor_box = self.anchor_box

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'single_object_dict_stack': single_label_dict,
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
            'past_k_time_diffs': past_k_time_diffs_stack})

        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})
        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(projected_lidar_stack)})

        return processed_data_dict


    def get_item_single_car(self, selected_cav_base, ego_pose, idx):
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
                object_bbx_center:		# ego view label. np.ndarray. Shape is (max_num, 7)
                object_ids:				# ego view label index. list. length is (max_num, 7)
                'curr_pose':			# current pose, list, len = 6
                'past_k_poses': 		# list of past k frames' poses
                'past_k_features': 		# list of past k frames' lidar
                'past_k_time_diffs': 	# list of past k frames' time diff with current frame
                'past_k_tr_mats': 		# list of past k frames' transformation matrix to current ego coordinate
                'past_k_label_dicts'    # [], [TBD] list of past k frames' label dict 
            }
        """
        selected_cav_processed = {}

        # curr lidar feature
        lidar_np = selected_cav_base['curr']['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np) # remove points that hit itself

        if self.visualize:
            # trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['curr']['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
            selected_cav_processed.update(
                {
                    'projected_lidar': projected_lidar
                }
            )

        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        curr_feature = self.pre_processor.preprocess(lidar_np)
        
        # past k transfomation matrix
        past_k_tr_mats = []
        # past k lidars
        past_k_features = []
        # past k poses
        past_k_poses = []
        # past k timestamps
        past_k_time_diffs = []
        
        # past k label 
        past_k_label_dicts = [] # TODO: 这个部分可以删掉

        # past k frames [trans matrix], [lidar feature], [pose], [time interval]
        for i in range(self.k):
            # 1. trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray
            past_k_tr_mats.append(transformation_matrix)

            # 2. lidar feature
            lidar_np = selected_cav_base['past_k'][i]['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np) # remove points that hit itself
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            processed_features = self.pre_processor.preprocess(lidar_np)
            past_k_features.append(processed_features)

            # 3. pose
            past_k_poses.append(selected_cav_base['past_k'][i]['params']['lidar_pose'])

            # 4. time interval
            if selected_cav_base['ego']:
                past_k_time_diffs.append(-i)
            else:
                past_k_time_diffs.append(-(self.tau+i))

            ################################################################
            # sizhewei
            # for past k frames' single view label
            ################################################################
            '''
            # 5. single view label
            # past_i label at past_i single view
            # opencood/data_utils/post_processor/base_postprocessor.py
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][i]['params']['lidar_pose'])  
            # generate the anchor boxes
            # opencood/data_utils/post_processor/voxel_postprocessor.py
            anchor_box = self.anchor_box
            single_view_label_dict = self.post_processor.generate_label(
                    gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
                )
            past_k_label_dicts.append(single_view_label_dict)
            '''

        '''
        # past k merge
        past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts)
        '''

        past_k_tr_mats = np.stack(past_k_tr_mats, axis=0) # (k, 4, 4)

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
        
        # curr label at ego view
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([selected_cav_base['curr']], ego_pose)
            
        selected_cav_processed.update(
            {"single_label_dict": label_dict,
             "curr_feature": curr_feature,
             'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'curr_pose': selected_cav_base['curr']['params']['lidar_pose'],
             'past_k_tr_mats': past_k_tr_mats,
             'past_k_poses': past_k_poses,
             'past_k_features': past_k_features,
             'past_k_time_diffs': past_k_time_diffs,
             'past_k_label_dicts': past_k_label_dicts
             })

        return selected_cav_processed

    @staticmethod
    def return_timestamp_key_async(cav_content, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # retrieve the correct index
        timestamp_key = list(cav_content.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def dist_time(ts1, ts2, i = -1):
        """caculate the time interval between two timestamps

        Args:
            ts1 (string): time stamp at some time
            ts2 (string): current time stamp
            i (int, optional): past frame id, for debug use. Defaults to -1.
        
        Returns:
            time_diff (float): time interval (ts1 - ts2)
        """
        if not i==-1:
            return -i
        else:
            return (int(ts1) - int(ts2))

    def merge_past_k_features_to_dict(self, processed_feature_list):
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

        for cav_id in range(len(processed_feature_list)):
            for time_id in range(self.k):
                for feature_name, feature in processed_feature_list[cav_id][time_id].items():
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
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

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
            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            # past_k_label_list.append(ego_dict['past_k_label_dicts'])

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
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({
            'single_object_label': single_object_label,
            'curr_processed_lidar': curr_processed_lidar_torch_dict,
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'processed_lidar': processed_lidar_torch_dict,
            'record_len': record_len,
            'label_dict': label_torch_dict,
            'single_past_dict': past_k_single_label_torch_dict,
            'object_ids': object_ids[0],
            'pairwise_t_matrix': pairwise_t_matrix,
            'curr_lidar_pose': curr_lidar_pose,
            'past_lidar_pose': past_k_lidar_pose,
            'past_k_time_interval': past_k_time_interval})

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
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({
            'transformation_matrix': transformation_matrix_torch,
            'transformation_matrix_clean': transformation_matrix_clean_torch})

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        # output_dict['ego'].update({'veh_frame_id': batch[0]['ego']['veh_frame_id']})

        return output_dict

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
