# -*- coding: utf-8 -*-
# Author: sizhewei @ 2023/4/15
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
two stage flow update framework

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
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange
# import opencood.data_utils.post_processor as post_processor
# import opencood.utils.pcd_utils as pcd_utils
# from opencood.utils.keypoint_utils import bev_sample, get_keypoints
# from opencood.data_utils.datasets import basedataset
# from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
# from opencood.utils.pcd_utils import \
#     mask_points_by_range, mask_ego_points, shuffle_points, \
#     downsample_lidar_minimum
# from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
# from opencood.utils.pose_utils import add_noise_data_dict, remove_z_axis
from opencood.utils.common_utils import read_json
# from opencood.utils import box_utils
# from opencood.utils.flow_utils import generate_flow_map, generate_flow_map_szwei

from opencood.utils.box_utils import boxes_to_corners2d # for debug use
# from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

# global, for debug use
illegal_path_list = set()


def extract_timestamps(yaml_files):
    """
    Given the list of the yaml files, extract the mocked timestamps.

    Parameters
    ----------
    yaml_files : list
        The full path of all yaml files of ego vehicle

    Returns
    -------
    timestamps : list
        The list containing timestamps only.
    """
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]
        if res.endswith('.yaml'):
            timestamp = res.replace('.yaml', '')
        elif res.endswith('.json'):
            timestamp = res.replace('.json', '')
        else:
            print("Woops! There is no processing method for file {}".format(res))
            sys.exit(1)
        timestamps.append(timestamp)

    return timestamps


def statistics_irv2v(split_name="validate"):
    # total box number
    box_total = 0
    # box number distribution
    box_num_dict = {}
    # speed distribution
    speed_dict = {}

    # 处理的数据集
    data_seperate_name = split_name
    # 文件路径
    root_dir = "/remote-home/share/OPV2V_irregular_npy/" + data_seperate_name

    # first load all paths of different scenarios
    scenario_folders = sorted([os.path.join(root_dir, x)
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])
    scenario_folders_name = sorted([x
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])

    print("### Processing Started! ###")

    # loop over all scenarios
    for (i, scenario_folder) in tenumerate(scenario_folders):
        start_time = time.time()
        # copy timestamps npy file
        timestamps_file = os.path.join(scenario_folder, 'timestamps.npy')
        time_annotations = np.load(timestamps_file)

        # save_scene_folder = os.path.join(save_dir, scenario_folders_name[i])
        # if not os.path.exists(save_scene_folder):
        #     os.makedirs(save_scene_folder)
        # save_timestamps_file = os.path.join(save_scene_folder, 'timestamps.npy')
        # if not os.path.exists(save_timestamps_file):
        #     shutil.copyfile(timestamps_file, save_timestamps_file)

        # iterate all cav in this scenario
        cav_list = sorted([x 
                            for x in os.listdir(scenario_folder) if 
                            os.path.isdir(os.path.join(scenario_folder, x))], key=lambda y:int(y))
        assert len(cav_list) > 0

        # use the frame number as key, the full path as the values, store all json or yaml files in this scenario
        yaml_files = sorted([x
                    for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if
                    x.endswith(".json")], 
                    key=lambda y:float((y.split('/')[-1]).split('.json')[0]))
        if len(yaml_files)==0:
            yaml_files = sorted([x 
                        for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if
                        x.endswith('.yaml')], key=lambda y:float((y.split('/')[-1]).split('.yaml')[0]))

        start_timestamp = int(float(extract_timestamps(yaml_files)[0]))
        while(1):
            time_id_json = ("%.3f" % float(start_timestamp)) + ".json"
            time_id_yaml = ("%.3f" % float(start_timestamp)) + ".yaml"
            if not (time_id_json in yaml_files or time_id_yaml in yaml_files):
                start_timestamp += 1
            else:
                break

        end_timestamp = int(float(extract_timestamps(yaml_files)[-1]))
        if start_timestamp%2 == 0:
            # even
            end_timestamp = end_timestamp-1 if end_timestamp%2==1 else end_timestamp
        else:
            end_timestamp = end_timestamp-1 if end_timestamp%2==0 else end_timestamp
        num_timestamps = int((end_timestamp - start_timestamp)/2 + 1)
        regular_timestamps = [start_timestamp+2*i for i in range(num_timestamps)]

        for (j, cav_id) in tenumerate(cav_list):
            cav_path = os.path.join(scenario_folder, cav_id)
            if j==0: # ego
                timestamps = regular_timestamps
            else: 
                timestamps = list(time_annotations[j-1, :])

            for timestamp in timestamps:
                timestamp = "%.3f" % float(timestamp)
                
                # self.scenario_database[i][cav_id][timestamp] = OrderedDict()

                json_file = os.path.join(cav_path,
                                            timestamp + '.json')
                json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
                
                with open(json_file, "r") as f:
                    curr_label = json.load(f)
            
                scene_box_num = len(curr_label['vehicles'].keys())
                # total box
                box_total += scene_box_num
                # box number distribution
                if scene_box_num in box_num_dict.keys():
                    box_num_dict[scene_box_num] += 1
                else:
                    box_num_dict[scene_box_num] = 1
                # speed distribution
                for car_idx, car_content in curr_label['vehicles'].items():
                    if int(car_content['speed']) in speed_dict.keys():
                        speed_dict[int(car_content['speed'])] += 1
                    else:
                        speed_dict[int(car_content['speed'])] = 1
    
    print(f"box_total = {box_total}")
    save_path = '/root/percp/OpenCOOD/opencood/'
    torch.save(box_num_dict, f"{save_path}box_num_dict_{split_name}.pt")
    torch.save(speed_dict, f"{save_path}speed_dict_{split_name}.pt")

def speed_hist():
    # import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    # 假设你有一个名为 dict 的字典
    split_name_list = ['train', 'validate', 'test']
    dicts = []
    for split_name in split_name_list:
        speed_dict_path = f"/root/percp/OpenCOOD/opencood/speed_dict_{split_name}.pt"
        unit_dict = torch.load(speed_dict_path, map_location='cpu')
        dicts.append(unit_dict)

    result_dict = {}
    for unit_dict in dicts:
        for key, value in unit_dict.items():
            if key in result_dict.keys():
                result_dict[key] += value
            else:
                result_dict[key] = value

    # 按照键的大小进行排序，生成排序后的键列表
    sorted_keys = sorted(result_dict.keys())

    # 根据排序后的键列表，获取对应的值列表
    sorted_values = [result_dict[key] for key in sorted_keys]

    # 绘制直方图
    plt.bar(sorted_keys[1:], sorted_values[1:])

    avg_speed = np.dot(np.array(sorted_keys),np.array(sorted_values)) / np.sum(np.array(sorted_values[1:]))
    print(f'avg speed is {avg_speed}')
    print(f'max speed is {sorted_keys[-1]}')
    print(f'speed=0 has cav: {sorted_values[0]}')
    print(f'all cav num is {np.sum(np.array(sorted_values))}')

    # labels = []
    # for i, unit_dict in enumerate(dicts):
    #     # 按照键的大小进行排序，生成排序后的键列表
    #     sorted_keys = sorted(unit_dict.keys())

    #     # 根据排序后的键列表，获取对应的值列表
    #     sorted_values = [unit_dict[key] for key in sorted_keys]

    #     # 绘制直方图
    #     plt.bar(sorted_keys[1:], sorted_values[1:], label=split_name_list[i])

    #     avg_speed = np.dot(np.array(sorted_keys),np.array(sorted_values)) / np.sum(np.array(sorted_values[1:]))
    #     print(f'{split_name} avg speed is {avg_speed}')

    #     labels.append(split_name_list[i])

    # 设置图表标题和坐标轴标签
    # plt.title('speed distribution histogram.')

    fontsize = 20
    # 设置图表标题和坐标轴标签
    # plt.title('num of box distribution histogram.')
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.xlabel('Speed of the vehicle (km/h)', fontsize=fontsize, color='black')
    plt.ylabel('Number of vehicles', fontsize=fontsize, color='black')
    plt.tight_layout()

    # plt.legend(labels)

    # 展示图表
    plt.savefig(f'/root/percp/OpenCOOD/opencood/distribution-speed.png', dpi=600)

def box_hist():
    # import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    # 假设你有一个名为 dict 的字典
    split_name_list = ['train', 'validate', 'test'] # ['test'] #
    dicts = []
    for split_name in split_name_list:
        speed_dict_path = f"/root/percp/OpenCOOD/opencood/box_num_dict_{split_name}.pt"
        unit_dict = torch.load(speed_dict_path, map_location='cpu')
        dicts.append(unit_dict)

    result_dict = {}
    for unit_dict in dicts:
        for key, value in unit_dict.items():
            if key in result_dict.keys():
                result_dict[key] += value
            else:
                result_dict[key] = value

    # 按照键的大小进行排序，生成排序后的键列表
    sorted_keys = sorted(result_dict.keys())

    # 根据排序后的键列表，获取对应的值列表
    sorted_values = [result_dict[key] for key in sorted_keys]

    # 绘制直方图
    plt.bar(sorted_keys, sorted_values)

    avg_speed = np.dot(np.array(sorted_keys),np.array(sorted_values)) / np.sum(np.array(sorted_values))
    print(f'avg num of box is {avg_speed}')
    print(f'max num is {sorted_keys[-1]}')
    print(f'total cav is : {np.sum(np.dot(np.array(sorted_keys), np.array(sorted_values)))}')

    # labels = []
    # for i, unit_dict in enumerate(dicts):
    #     # 按照键的大小进行排序，生成排序后的键列表
    #     sorted_keys = sorted(unit_dict.keys())

    #     # 根据排序后的键列表，获取对应的值列表
    #     sorted_values = [unit_dict[key] for key in sorted_keys]

    #     # 绘制直方图
    #     plt.bar(sorted_keys[1:], sorted_values[1:], label=split_name_list[i])

    #     avg_speed = np.dot(np.array(sorted_keys),np.array(sorted_values)) / np.sum(np.array(sorted_values[1:]))
    #     print(f'{split_name} avg speed is {avg_speed}')

    #     labels.append(split_name_list[i])

    fontsize = 20
    # 设置图表标题和坐标轴标签
    # plt.title('num of box distribution histogram.')
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.xlabel('Number of vehicles in the scene', fontsize=fontsize, color='black')
    plt.ylabel('Number of scenes', fontsize=fontsize, color='black')
    plt.tight_layout()
    # plt.legend(labels)

    # 展示图表
    plt.savefig(f'/root/percp/OpenCOOD/opencood/distribution-box.png', dpi=600)


if __name__ == '__main__':
    # statistics_irv2v()
    # speed_hist()
    box_hist()
    