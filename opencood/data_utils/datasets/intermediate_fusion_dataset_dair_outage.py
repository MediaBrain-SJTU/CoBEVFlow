# -*- coding: utf-8 -*-
# Modified: Sizhe Wei <sizhewei@sjtu.edu.cn>

"""
Dataset class for intermediate fusion with time delay k
"""
import random
import math
from collections import OrderedDict

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils

from opencood.data_utils.datasets import intermediate_fusion_dataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.pose_utils import add_noise_data_dict

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_idx_to_co_info(data):
    idx2info = {}
    for elem in data:
        if elem["vehicle_pointcloud_path"] == "":
            continue
        idx = elem["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

class IntermediateFusionDatasetDAIR_outage(intermediate_fusion_dataset.IntermediateFusionDataset):
    """
    Written by sizhewei @ 2022/09/28
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        #注意yaml文件应该有sensor_type：lidar/camera
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        self.k = params['num_of_past']
        
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = load_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False

        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
            
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None


        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        #这里root_dir是一个json文件！--> 代表一个split
        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = '/GPFS/rhome/quanhaoli/workspace/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure'
        self.inf_idx2info = build_idx_to_info(
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
        )
        self.co_idx2info = build_idx_to_co_info(
            load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )

        self.data_split = load_json(split_dir)
        self.data = []
        for veh_idx in self.data_split:
            if self.is_valid_id(veh_idx):
                self.data.append(veh_idx)
        if self.train:
            np.save('./train_list.npy',self.data,allow_pickle=True)
        else:
            np.save('./val_list.npy',self.data,allow_pickle=True)

        print("ASync dataset with {} time delay initialized! {} samples totally!".format(self.k, len(self.data)))
    def is_valid_id(self, veh_frame_id):
        """
        Written by sizhewei @ 2022/10/05
        Given veh_frame_id, determine whether there is a corresponding inf_frame that meets the k delay requirement.
        Modified by Shunli Ren @ 2022/11/3 
        judge it according to vehicle id
        
        Parameters
        ----------
        veh_frame_id : 05d
            Vehicle frame id

        Returns
        -------
        bool valud
            True means there is a corresponding road-side frame.
        """
        # print('veh_frame_id: ',veh_frame_id,'\n')
        frame_info = {}
        
        for i in range(self.k):
            past_id = id_to_str(int(veh_frame_id) - self.k + i)
            if past_id not in self.data_split:
                return False
        vehicle_idx_0 = id_to_str(int(veh_frame_id) - self.k)

            
        frame_info = self.co_idx2info[veh_frame_id]
        frame_info_idx0 = self.co_idx2info[vehicle_idx_0]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        inf_frame_id_idx0 = frame_info_idx0['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        cur_inf_info = self.inf_idx2info[inf_frame_id]
        if (
            int(inf_frame_id_idx0) < int(cur_inf_info["batch_start_id"])
        ):
            return False

        return True
    
    def get_vehicle_trans(self, veh_frame_id):

        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)
        trans = tfm_to_pose(transformation_matrix)

        return trans
    
    def get_inf_trans(self, inf_frame_id, system_error_offset):
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        trans = tfm_to_pose(transformation_matrix1)

        return trans

    def retrieve_base_data(self, idx):
        """
        Modified by sizhewei @ 2022/09/28
        Given the index, return the corresponding async data (time delay: self.k).

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.data[idx]
        # print('veh_frame_id: ',veh_frame_id,'\n')
        # frame_info = {}
        # system_error_offset = {}
        
        frame_info = self.co_idx2info[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        # inf_frame_id = id_to_str(int(inf_frame_id))
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        #cav_id=0是车端，1是路边单元
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        # print(data[0]['params']['vehicles'])
        data[0]['params']['lidar_pose'] = self.get_vehicle_trans(veh_frame_id)
        # lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        # novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        # transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)
        # data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]
            
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[1]['params'] = OrderedDict()

        # data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        data[1]['params']['vehicles'] = [] # we only load cooperative label in vehicle side

        data[1]['params']['lidar_pose'] = self.get_inf_trans(inf_frame_id,system_error_offset)
        # virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        # transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        # data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)


        cur_inf_info = self.inf_idx2info[inf_frame_id]
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir, \
            'infrastructure-side', cur_inf_info['pointcloud_path']))
        data[0]['veh_frame_id'] = veh_frame_id
        data[1]['veh_frame_id'] = inf_frame_id

        past_features = []

        for i in range(self.k):
            past_feature = OrderedDict()
            veh_id = id_to_str(int(veh_frame_id) - self.k + i)
            frame_info = self.co_idx2info[veh_id]
            inf_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
            past_feature['veh'] = OrderedDict()
            past_feature['inf'] = OrderedDict()
            system_error_offset = frame_info["system_error_offset"]
            past_feature['veh']['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            past_feature['veh']['lidar_pose'] = self.get_vehicle_trans(veh_id)

            past_feature['inf']['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
            past_feature['inf']['lidar_pose'] = self.get_inf_trans(inf_id, system_error_offset)

            past_features.append(past_feature)
            
        return data, past_features

    def __getitem__(self, idx):
        # base_data_dict = self.retrieve_base_data(idx)
        base_data_dict, past_features = self.retrieve_base_data(idx)

        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break
            
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0


        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue


            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)


        ########## Added by Yifan Lu 2022.8.14 ##############
        # box align to correct pose.
        '''
        if self.box_align and str(idx) in self.stage1_result.keys():
            stage1_content = self.stage1_result[str(idx)]
            if stage1_content is not None:
                cav_id_list_stage1 = stage1_content['cav_id_list']
                
                pred_corners_list = stage1_content['pred_corner3d_np_list']
                pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
                uncertainty_list = stage1_content['uncertainty_np_list']
                uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
                stage1_lidar_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list_stage1]
                stage1_lidar_pose = np.array(stage1_lidar_pose_list)


                refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                stage1_lidar_pose, 
                                                                uncertainty_list=uncertainty_list, 
                                                                **self.box_align_args)
                stage1_lidar_pose[:,[0,1,4]] = refined_pose
                stage1_lidar_pose_refined_list = stage1_lidar_pose.tolist() # updated lidar_pose_list
                for cav_id, lidar_pose_refined in zip(cav_id_list_stage1, stage1_lidar_pose_refined_list):
                    if cav_id not in cav_id_list:
                        continue
                    idx_in_list = cav_id_list.index(cav_id)
                    lidar_pose_list[idx_in_list] = lidar_pose_refined
                    base_data_dict[cav_id]['params']['lidar_pose'] = lidar_pose_refined
        '''     

        past_processed_features_list = self.get_past_precessed_features_dair(past_features)

        for cav_id in cav_id_list:

            selected_cav_base = base_data_dict[cav_id]



            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose, 
                ego_lidar_pose_clean,
                idx)
                
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

            processed_features.append(
                selected_cav_processed['processed_features'])
            if self.kd_flag:
                projected_lidar_clean_list.append(
                    selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                if cav_id== 0:
                    projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

                    # projected_lidar_stack.append(
                    # selected_cav_processed['projected_lidar'])





        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)
        
        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################

        ############ for disconet ###########
        if self.kd_flag:
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)

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
        cav_num = len(processed_features)

        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)
        
        temporal_trans_matrix = self.get_temporal_trans_matrix(past_features, base_data_dict[0]['params']['lidar_pose'])

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses,
             'past_features': past_processed_features_list,
             'temporal_trans_matrix': temporal_trans_matrix})


        

        if self.kd_flag:
            processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})


        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})

        # processed_data_dict['ego'].update({'veh_frame_id': base_data_dict[0]['veh_frame_id']})

        return processed_data_dict

    def __len__(self):
        # 符合条件的 frame 的数量
        return len(self.data)

    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
