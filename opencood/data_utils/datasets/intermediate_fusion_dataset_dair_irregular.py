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
from scipy import stats

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

def build_inf_fid_to_veh_fid(data):
    inf_fid2veh_fid = {}
    for elem in data:
        veh_fid = elem["vehicle_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid = elem["infrastructure_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid2veh_fid[inf_fid] = veh_fid
    return inf_fid2veh_fid

def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

class IntermediateFusionDatasetDAIRIrregular(intermediate_fusion_dataset.IntermediateFusionDataset):
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
        
        if 'num_sweep_frames' in params:    # number of frames we use in LSTM
            self.k = params['num_sweep_frames']
        else:
            self.k = 1

        if 'binomial_n' in params:
            self.binomial_n = params['binomial_n']
        else:
            self.binomial_n = 10

        if 'binomial_p' in params:
            self.binomial_p = params['binomial_p']
        else:
            self.binomial_p = 0
        
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
        self.anchor_box = self.post_processor.generate_anchor_box()

        #这里root_dir是一个json文件！--> 代表一个split
        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = '/remote-home/share/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure' # TODO:
        self.inf_idx2info = build_idx_to_info(
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
        )
        self.co_idx2info = build_idx_to_co_info(
            load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )
        self.inf_fid2veh_fid = build_inf_fid_to_veh_fid(load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )

        self.data_split = load_json(split_dir)
        self.data = []
        for veh_idx in self.data_split:
            if self.is_valid_id(veh_idx):
                self.data.append(veh_idx)

        print("Irregular async dataset with past %d frames and expectation time delay = %d initialized! %d samples totally!" % (self.k, int(self.binomial_n*self.binomial_p), len(self.data)))

    def is_valid_id(self, veh_frame_id):
        """
        Written by sizhewei @ 2022/10/05
        Given veh_frame_id, determine whether there is a corresponding inf_frame that meets the k delay requirement.

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
        
        frame_info = self.co_idx2info[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        cur_inf_info = self.inf_idx2info[inf_frame_id]
        if (int(inf_frame_id) - self.binomial_n*self.k < int(cur_inf_info["batch_start_id"])):
            return False
        for i in range(3*self.binomial_n * self.k):# TODO: delete 3
            delay_id = id_to_str(int(inf_frame_id) - i) 
            if delay_id not in self.inf_fid2veh_fid.keys():
                return False

        return True

    def retrieve_base_data(self, idx):
        """
        Modified by sizhewei @ 2022/09/28
        Given the index, return the corresponding async data (time delay expection = sum(B(n, p)) ).

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            {
                [0]  / [1]: {
                    'ego' : True / False,
                    'params' : {
                        'vehicles':
                        'lidar_pose':
                    },
                    'lidar_np' : 
                    'veh_frame_id' :
                    'avg_time_delay' : ([1] only)
                }
            }
        """
        veh_frame_id = self.data[idx]
        # print('veh_frame_id: ',veh_frame_id,'\n')
        frame_info = {}
        system_error_offset = {}
        
        frame_info = self.co_idx2info[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        # 生成冻结分布函数
        bernoulliDist = stats.bernoulli(self.binomial_p) 
        # B(n, p)
        trails = bernoulliDist.rvs(self.binomial_n)
        sample_interval = sum(trails)
        
        curr_inf_frame_id = inf_frame_id
        inf_frame_id = id_to_str(int(inf_frame_id) - sample_interval)

        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        #cav_id=0是车端，1是路边单元
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        # print(data[0]['params']['vehicles'])
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]

        data[0]['lidar_np_curr'] = data[0]['lidar_np'] # vehicle side lidar is no delay
            
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[1]['params'] = OrderedDict()

        # data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        data[1]['params']['vehicles'] = [] # we only load cooperative label in vehicle side

        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))

        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)


        # single label 
        data[0]['params']['vehicles_single'] = load_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single'] = load_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))

        cur_inf_info = self.inf_idx2info[inf_frame_id]
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir, \
            'infrastructure-side', cur_inf_info['pointcloud_path']))

        curr_inf_info = self.inf_idx2info[curr_inf_frame_id]
        data[1]['lidar_np_curr'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir, \
            'infrastructure-side', curr_inf_info['pointcloud_path']))

        data[0]['veh_frame_id'] = veh_frame_id
        data[1]['veh_frame_id'] = inf_frame_id
        data[1]['avg_time_delay'] = sample_interval
        return data


    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean, idx):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation
        
        idx: int,
            debug use.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}



        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose) # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                     ego_pose_clean)

        # retrieve objects under ego coordinates
        # this is used to generate accurate GT bounding box.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                    ego_pose_clean)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)

        # current lidar np, w.o. delay
        # filter lidar
        curr_lidar_np = selected_cav_base['lidar_np_curr']
        curr_lidar_np = shuffle_points(curr_lidar_np)
        # remove points that hit itself
        curr_lidar_np = mask_ego_points(curr_lidar_np)

        # project the lidar to ego space
        # x,y,z in ego space
        projected_lidar = \
            box_utils.project_points_by_matrix_torch(curr_lidar_np[:, :3],
                                                        transformation_matrix)
        if self.kd_flag:
            lidar_np_clean = copy.deepcopy(lidar_np)

        if self.proj_first:
            lidar_np[:, :3] = projected_lidar
            
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': projected_lidar,
             'processed_features': processed_lidar,
             'transformation_matrix': transformation_matrix,
             'transformation_matrix_clean': transformation_matrix_clean})

        if self.kd_flag:
            projected_lidar_clean = \
                box_utils.project_points_by_matrix_torch(lidar_np_clean[:, :3],
                                                    transformation_matrix_clean)
            lidar_np_clean[:, :3] = projected_lidar_clean
            lidar_np_clean = mask_points_by_range(lidar_np_clean,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
            selected_cav_processed.update(
                {"projected_lidar_clean": lidar_np_clean}
            )

        # generate targets label single GT, note the reference pose is itself.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
            [selected_cav_base], selected_cav_base['params']['lidar_pose']
        )
        
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({
                            "single_label_dict": label_dict,
                            "single_object_bbx_center": object_bbx_center,
                            "single_object_bbx_mask": object_bbx_mask})

        return selected_cav_processed

    def __getitem__(self, idx):
        # base_data_dict = self.retrieve_base_data(idx)
        base_data_dict = self.retrieve_base_data(idx)

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
        single_label_list = []
        single_object_bbx_center_list = []
        single_object_bbx_mask_list = []

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
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

            single_label_list.append(selected_cav_processed['single_label_dict'])
            single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
            single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

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

        # single label
        single_label_dicts = self.post_processor.collate_batch(single_label_list)
        single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
        single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
        processed_data_dict['ego'].update({
            "single_label_dict_torch": single_label_dicts,
            "single_object_bbx_center_torch": single_object_bbx_center,
            "single_object_bbx_mask_torch": single_object_bbx_mask,
            })


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
             'lidar_poses': lidar_poses})

        # for irregular time delay use:
        if len(too_far) == 0:
            avg_time_delay = base_data_dict[1]['avg_time_delay']
            cp_rate = 1
        else:
            avg_time_delay = self.binomial_n
            cp_rate = 0
        processed_data_dict['ego'].update(
            {'avg_time_delay': avg_time_delay,
                'cp_rate': cp_rate})

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

    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"

        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        pos_equal_one_single = []
        neg_equal_one_single = []
        targets_single = []
        object_bbx_center_single = []
        object_bbx_mask_single = []


        # average time delay
        avg_time_delay = []
        # cp rate
        cp_rate = []
        
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        if self.kd_flag:
            teacher_processed_lidar_list = []
        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])

            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            avg_time_delay.append(ego_dict['avg_time_delay'])
            cp_rate.append(ego_dict['cp_rate'])

            if self.kd_flag:
                teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

            pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
            neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
            targets_single.append(ego_dict['single_label_dict_torch']['targets'])
            object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
            object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))


        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)

        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)  # coords 增加了 batch_id 维度，[x, y, z] -> [id, x, y, z]
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'lidar_pose': lidar_pose})

        avg_time_delay = float(np.mean(np.array(avg_time_delay)))
        cp_rate = float(np.mean(np.array(cp_rate)))
        output_dict['ego'].update({'avg_time_delay': avg_time_delay,
                                      'cp_rate': cp_rate})  

        output_dict['ego'].update({
            "single_object_label":{ # the same name as opencood/data_utils/datasets/intermediate_fusion_dataset_opv2v_irregular_flow_new.py 
                    "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                    "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                    "targets": torch.cat(targets_single, dim=0),
                    # for centerpoint
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                },
            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
        })


        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})
        
        if self.kd_flag:
            teacher_processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(teacher_processed_lidar_list)
            output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})

        if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and \
            (output_dict['ego']['processed_lidar']['voxel_coords'][:, 0].max().int().item() + 1) != record_len.sum().int().item():
            return None

        return output_dict
