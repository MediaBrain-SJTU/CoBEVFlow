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
# import open3d as o3d
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

    root_dirs = "/DB/data/sizhewei/logs"
    save_path = './opencood/result.jpg'
    
    split_list = ['where2comm_irregular', 'latefusion_irregular']
    single_split_name = 'single_irregular'

    num_delay = 10

    max_x = 1000 # unit is ms
    plt.figure()
    fig, ax = plt.subplots(3,1, sharex='col', sharey=False, figsize=(6,9))
    fig.text(0.5, 0.06, '# Avg. time delay (ms)', ha='center', fontsize='x-large')
    fig.text(0.06, 0.5, 'AP', va='center', rotation='vertical', fontsize='x-large')
    ax30 = ax[0]; ax50 = ax[1]; ax70 = ax[2]

    for split_name in split_list:
        ap_list = []
        delays = []
        for i in trange(num_delay+1):
            log_file = split_name + '_d_' + str(i)
            eval_file = os.path.join(root_dirs, log_file, "eval_no_noise_ir_in_ir_gt_update.yaml")
            tmp_aps = []
            with open(eval_file, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            tmp_aps.append(data['ap_30'])
            tmp_aps.append(data['ap_50'])
            tmp_aps.append(data['ap_70'])
            try:
                delays.append(-data['avg_time_delay'])
            except:
                delays.append(i)

            ap_list.append(tmp_aps)

        ap_list_np = np.array(ap_list)
        ap_list_np = np.transpose(ap_list_np)

        ap_30_list = list(ap_list_np[0])
        ap_50_list = list(ap_list_np[1])
        ap_70_list = list(ap_list_np[2])
        method_name = split_name.split('_')[0]

        color = '#1f77b4' if split_name==split_list[0] else 'g'
        plt.sca(ax30); plt.plot(delays, ap_30_list, color=color, marker='+', label = method_name + '_' + 'ap30')
        plt.sca(ax50); plt.plot(delays, ap_50_list, color=color, marker='+', label = method_name + '_' + 'ap50')
        plt.sca(ax70); plt.plot(delays, ap_70_list, color=color, marker='+', label = method_name + '_' + 'ap70')
    
    # for single fusion
    method_name = single_split_name.split('_')[0]
    eval_file = os.path.join(root_dirs, single_split_name, 'eval_no_noise_ir_in_ir_gt_update.yaml')
    with open(eval_file, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    plt.sca(ax30); plt.plot([0,max_x],[data['ap_30'],data['ap_30']], 'r--', label=method_name + '_' + 'ap30')
    plt.sca(ax50); plt.plot([0,max_x],[data['ap_50'],data['ap_50']], 'r--', label=method_name + '_' + 'ap50')
    plt.sca(ax70); plt.plot([0,max_x],[data['ap_70'],data['ap_70']], 'r--', label=method_name + '_' + 'ap70')

    xaxis = np.linspace(0, max_x, 11)
    ax30.set_title('The Results for AP@0.3'); ax30.grid(True); 
    ax30.set_xticks(xaxis); \
        ax30.legend(loc = 'upper right')
    ax50.set_title('The Results for AP@0.5'); ax50.grid(True); 
    ax50.set_xticks(xaxis); \
        ax50.legend(loc = 'upper right')
    ax70.set_title('The Results for AP@0.7'); ax70.grid(True); 
    ax70.set_xticks(xaxis); \
        ax70.legend(loc = 'upper right')

    plt.savefig(save_path)
    print("=== Plt save finished!!! ===")
    # plt.title('标题')
    # plt.show()
