# transfer all lidar file from .pcd to .npy 
# sizhewei @ <sizhewei@sjtu.edu.cn>
# 2022/12/20

###################
# $ cd ~/percp/OpenCOOD/
# python opencood/utils/my_tools.py
####################

import os
import sys
sys.path.append(os.getcwd())
import shutil
import open3d as o3d
import numpy as np
import time
import json
import yaml
from pypcd import pypcd
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange

import matplotlib.pyplot as plt

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import pcd_to_np

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

        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps

def to_npy_and_json(split_name="train"):
    # 处理的数据集
    data_seperate_name = split_name
    # pcd 文件路径
    root_dir = "/GPFS/public/OPV2V_Irregular_V2/" + data_seperate_name
    # yaml 文件路径
    root_annotation_dir = "/GPFS/public/OPV2V_Irregular_V2/dataset_irregular_v2/" + data_seperate_name
    # 新保存的路径
    save_dir = "/GPFS/public/OPV2V_irregular_npy/" + data_seperate_name

    # first load all paths of different scenarios
    scenario_folders = sorted([os.path.join(root_dir, x)
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])
    scenario_folders_name = sorted([x
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])

    print("### Transformation Started! ###")

    # loop over all scenarios
    for (i, scenario_folder) in tenumerate(scenario_folders):
        start_time = time.time()
        # copy timestamps npy file
        timestamps_file = os.path.join(root_annotation_dir, scenario_folders_name[i], 'timestamps.npy')
        save_scene_folder = os.path.join(save_dir, scenario_folders_name[i])
        if not os.path.exists(save_scene_folder):
            os.makedirs(save_scene_folder)
        save_timestamps_file = os.path.join(save_scene_folder, 'timestamps.npy')
        if not os.path.exists(save_timestamps_file):
            shutil.copyfile(timestamps_file, save_timestamps_file)

        # iterate all cav in this scenario
        cav_list = sorted([x 
                            for x in os.listdir(scenario_folder) if 
                            os.path.isdir(os.path.join(scenario_folder, x))])
        assert len(cav_list) > 0
        for (j, cav_id) in tenumerate(cav_list):
            cav_path_pcd = os.path.join(scenario_folder, cav_id)
            cav_path_yaml = os.path.join(root_annotation_dir, scenario_folders_name[i], cav_id)
            # use the frame number as key, the full path as the values
            yaml_files = \
                sorted([os.path.join(cav_path_yaml, x)
                        for x in os.listdir(cav_path_yaml) if
                        x.endswith('.yaml')])
            timestamps = extract_timestamps(yaml_files)	

            for k, timestamp in tenumerate(timestamps):
                # if k > 30:
                #     break

                # new save dir
                save_path = os.path.join(save_dir, scenario_folders_name[i], cav_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    # print("======Path {} created successfully!======".format(save_path))

                # change pcd to npy 
                lidar_file = os.path.join(cav_path_pcd, timestamp + '.pcd')
                new_lidar_file = os.path.join(save_path, timestamp + '.npy')
                if not os.path.exists(new_lidar_file):
                    lidar = pcd_to_np(lidar_file)
                    np.save(new_lidar_file, lidar)

                # change yaml to json
                yaml_file = os.path.join(cav_path_yaml, timestamp + '.yaml')
                json_file = os.path.join(cav_path_yaml, timestamp + '.json')
                new_json_file = os.path.join(save_path, timestamp + '.json')
                if os.path.exists(json_file): # 原数据集有json文件
                    if not os.path.exists(new_json_file):
                        shutil.copyfile(json_file, new_json_file)
                else:
                    if not os.path.exists(new_json_file):
                        params = load_yaml(yaml_file)
                        json_str = json.dumps(params, indent=4)
                        with open(new_json_file, 'w') as json_file:
                            json_file.write(json_str)
                
                # new_yaml_file = os.path.join(save_path, timestamp + '.yaml')
                # if not os.path.exists(new_yaml_file):
                #     shutil.copyfile(yaml_file, new_yaml_file)

                # print(timestamp)
        end_time = time.time()
        print("=== %d-th scenario (%s) finished, consumed %.3f mins. ===" \
            % (i, scenario_folders_name[i], (end_time-start_time)/60.0))

def create_small_dataset(split_name='train', size_scene=4, size_cav=3, size_time=2):
    """
    TO BE FINISHED !!!
    Create a small dataset from original large one 
    """
    # split name
    data_seperate_name = split_name
    # 原数据的路径
    root_dir = "/GPFS/public/OPV2V_irregular_npy/" + data_seperate_name
    # 新保存的路径
    save_dir = "/GPFS/public/OPV2V_irregular_npy_small/" + data_seperate_name

    # first load all paths of different scenarios
    scenario_folders = sorted([os.path.join(root_dir, x)
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])
    scenario_folders_name = sorted([x
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])

    print("### Transformation Started! ###")

    # loop over all scenarios
    for (i, scenario_folder) in tenumerate(scenario_folders):
        if i==size_scene:
            break

if __name__ == "__main__":
    # to_npy_and_json(split_name='validate')
    # create_small_dataset(split_name='train')

    log_dirs = "/DB/data/sizhewei/logs/opv2v_npj_raindrop_max_multiscale_resnet_d_0_swps_w_resnet_w_multiscale_for_inference_regular_d_"
    save_path = './opencood/ap_delay_regular.jpg'

    ap_list = []
    for i in trange(11):
        log_file = log_dirs + str(i)
        eval_file = os.path.join(log_file, "eval_no_noise.yaml")
        tmp_aps = []
        with open(eval_file, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        tmp_aps.append(data['ap_30'])
        tmp_aps.append(data['ap_50'])
        tmp_aps.append(data['ap_70'])

        ap_list.append(tmp_aps)

    ap_list_np = np.array(ap_list)
    ap_list_np = np.transpose(ap_list_np)

    delays = list(range(11))
    ap_30_list = list(ap_list_np[0])
    ap_50_list = list(ap_list_np[1])
    ap_70_list = list(ap_list_np[2])
    plt.plot(delays, ap_30_list, label='ap30')
    plt.plot(delays, ap_50_list, label='ap50')
    plt.plot(delays, ap_70_list, label='ap70')
    
    plt.axis([0, 11, 0, 1])
    plt.xlabel('# delay timestamps')
    plt.ylabel('AP')
    plt.legend(loc = 'upper right')
    plt.savefig(save_path)
    print("=== Plt save finished!!! ===")
    # plt.title('标题')
    plt.show()
