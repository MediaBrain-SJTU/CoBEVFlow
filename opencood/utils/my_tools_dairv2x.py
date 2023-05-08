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

# 用于单独保存子图的函数
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path+fig_name, bbox_inches=extent, dpi=600, format='pdf')

if __name__ == "__main__":
    # to_npy_and_json(split_name='validate')
    # create_small_dataset(split_name='train')

    root_dirs = "/remote-home/share/sizhewei/logs"
    note = 'major_dairv2x'
    format = 'pdf'
    save_path = f"./opencood/result_{note}.{format}"
    title = "Average Precision curves of different methods on the IRV2V dataset at different average time intervals."
    
    split_list = ['late', 'v2vnet', 'v2xvit', 'disconet', 'where2comm', 'where2comm_syncnet', 'cobevflow'] #
    colors = ['#999999', '#E7DAD2', '#BEB8DC', '#82B0D2', '#8ECFC9', '#FFBE7A', '#FA7F6F']
    # colors = ['lightskyblue', 'lightseagreen', 'tomato', 'orange', 'gray', 'purple']
    single_split_name = 'single'

    num_delay = 15

    max_x = 500 # unit is ms
    plt.figure()
    fig, ax = plt.subplots(1,2, sharex='col', sharey=False, figsize=(18,6))
    fig.suptitle(f'{title}', fontsize='x-large', y=0.99)
    # fig.text(0.5, 0.03, 'Expectation of time intervals (ms).', ha='center', fontsize='x-large')
    # fig.text(0.08, 0.5, 'AP', va='center', rotation='vertical', fontsize='x-large')
    # ax30 = ax[0]; 
    ax50 = ax[0]; ax70 = ax[1]

    # for single fusion
    method_name = 'Single'
    # eval_file = os.path.join(root_dirs, f'eval_{single_split_name}.yaml')
    eval_file = '/remote-home/share/sizhewei/logs/dairv2x_no_fusion_ylu/eval_no_noise_no_all_0.00.yaml'
    single_color = 'black'
    with open(eval_file, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # plt.sca(ax30); plt.plot([0,max_x],[data['ap_30'],data['ap_30']], color=single_color, linestyle='--', label=method_name)
    plt.sca(ax50); plt.plot([0,max_x],[data['ap_50'],data['ap_50']], color=single_color, linestyle='--', label=method_name)
    plt.sca(ax70); plt.plot([0,max_x],[data['ap_70'],data['ap_70']], color=single_color, linestyle='--', label=method_name)

    # for single fusion
    method_name = 'Where2comm (no fusion)'
    # eval_file = os.path.join(root_dirs, f'eval_{single_split_name}.yaml')
    eval_file = '/remote-home/share/sizhewei/logs/dairv2x_where2comm_baseline_ylu/eval_no_noise_where2comm_no_fusion_0.00_0.yaml' #'/remote-home/share/sizhewei/logs/dairv2x_no_fusion_ylu/eval_no_noise_no_all_0.00.yaml'
    single_color = 'red'
    with open(eval_file, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # plt.sca(ax30); plt.plot([0,max_x],[data['ap_30'],data['ap_30']], color=single_color, linestyle='--', label=method_name)
    plt.sca(ax50); plt.plot([0,max_x],[data['ap_50'],data['ap_50']], color=single_color, linestyle='-.', label=method_name)
    plt.sca(ax70); plt.plot([0,max_x],[data['ap_70'],data['ap_70']], color=single_color, linestyle='-.', label=method_name)

    for split_i, split_name in enumerate(split_list):
        ap_list = []
        delays = []
        if split_name == 'late':
            method_name = 'Late Fusion'
            file_name = 'dairv2x_late_fusion_yflu'
            eval_name = 'late'
        elif split_name == 'v2vnet':
            method_name = 'V2VNet'
            file_name = 'dairv2x_v2vnet'
            eval_name = 'v2vnet'
        elif split_name == 'v2xvit':
            method_name = 'V2X-ViT'
            file_name = 'dairv2x_v2xvit_ylu'
            eval_name = 'v2xvit'
        elif split_name == 'disconet':
            method_name = 'DiscoNet'
            file_name = 'dairv2x_disconet_sren'
            eval_name = 'disconet'
        elif split_name == 'where2comm':
            method_name = 'Where2comm_all'
            file_name = 'dairv2x_where2comm_baseline_ylu'
            eval_name = 'where2comm'
        elif split_name == 'where2comm_syncnet':
            method_name = 'Where2comm + SyncNet'
            file_name = 'dariv2x_where2comm_syncnet'
            eval_name = 'syncnet'
        elif split_name == 'cobevflow':
            method_name = 'CoBEVFlow (ours)'
            file_name = 'dairv2x_where2comm_cobevflow'
            eval_name = 'cobevflow'
        latest_time_delay = -100.00
        for i in tqdm(np.linspace(0, 0.5, 26)):
            if i < 0.1 and i != 0:
                continue
            # log_file = os.path.join(split_name, f"eval_{note}_%.1f.yaml" % i)
            eval_file = os.path.join(root_dirs, file_name, f"eval_no_noise_{eval_name}_%.2f.yaml" % i)
            # if split_name == 'cobevflow':
            #     eval_file = os.path.join('/remote-home/share/sizhewei/logs/irv2v_where2comm_cobevflow/eval_draw_curve', f"eval_no_noise_{eval_name}_%.2f.yaml" % i)
            
            if not os.path.exists(eval_file):
                print(f'eval file {eval_file} not exist!')
                continue
            with open(eval_file, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            
            # try:
            #     unit_time_delay = -data['avg_time_delay']
            # except:
            unit_time_delay = i*1000

            if (unit_time_delay - latest_time_delay) < 30:
                continue
            latest_time_delay = unit_time_delay
            
            tmp_aps = []
            tmp_aps.append(data['ap_30'])
            tmp_aps.append(data['ap_50'])
            tmp_aps.append(data['ap_70'])
            ap_list.append(tmp_aps)

            delays.append(unit_time_delay)

        if len(ap_list) == 0:
            continue

        ap_list_np = np.array(ap_list)
        ap_list_np = np.transpose(ap_list_np)

        ap_30_list = list(ap_list_np[0])
        ap_50_list = list(ap_list_np[1])
        ap_70_list = list(ap_list_np[2])
        
        color = colors[split_i]
        # plt.sca(ax30); plt.plot(delays, ap_30_list, color=color, marker='+', label = method_name)
        plt.sca(ax50); plt.plot(delays, ap_50_list, color=color, linewidth='2', label = method_name, marker='+') # 
        plt.sca(ax70); plt.plot(delays, ap_70_list, color=color, linewidth='2', label = method_name, marker='+') # , marker='+'


    xaxis = np.linspace(0, max_x, 6)
    # ax30.set_title('The Results for AP@0.3'); ax30.grid(True); 
    # ax30.set_xticks(xaxis); \
    #     ax30.legend(loc = 'lower left')
    yaxis = np.linspace(0.55, 0.85, 7)
    ax50.set_title('The Results for AP@0.5'); ax50.grid(True); 
    ax50.set_xticks(xaxis); ax50.set_yticks(yaxis); 
    ax50.set_xlabel('Expectation time delay of the latest frame (ms)'); ax50.set_ylabel('AP@0.50');
    ax50.legend(ncol=2, loc = 'best') # ax50.legend(loc = 'lower left')
    
    yaxis = np.linspace(0.4, 0.7, 7)
    ax70.set_title('The Results for AP@0.7'); ax70.grid(True); 
    ax70.set_xticks(xaxis); ax70.set_yticks(yaxis); 
    ax70.set_xlabel('Expectation time delay of the latest frame (ms)'); ax70.set_ylabel('AP@0.70');
    ax70.legend(ncol=2, loc = 'best') # ax70.legend(loc = 'lower left')

    save_subfig(fig,ax[0],'./opencood/','dairv2x_ap50.pdf')
    save_subfig(fig,ax[1],'./opencood/','dairv2x_ap70.pdf')

    plt.savefig(save_path, dpi=600, format=format)
    print("=== Plt save finished!!! ===")
    # plt.title('标题')
    # plt.show()
