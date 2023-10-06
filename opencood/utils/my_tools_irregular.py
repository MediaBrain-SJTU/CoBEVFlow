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

# 用于单独保存子图的函数
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path+fig_name, bbox_inches=extent, dpi=600, format='pdf')

if __name__ == "__main__":
    # to_npy_and_json(split_name='validate')
    # create_small_dataset(split_name='train')

    root_dirs = "/remote-home/share/sizhewei/logs"
    note = 'major_irv2v_irregular_f500'
    format = 'jpg'
    save_path = f"./opencood/result_{note}.{format}"
    title = "Average Precision curves of different methods on the IRV2V dataset at different average time intervals."
    
    split_list = ['where2comm_syncnet', 'cobevflow'] #['late', 'v2vnet', 'v2xvit', 'disconet', 'where2comm', 
    colors = ['#FFBE7A', '#FA7F6F'] #'#E7DAD2'['#999999', '#DEC68B', '#BEB8DC', '#82B0D2', '#8ECFC9', 
    # colors = ['lightskyblue', 'lightseagreen', 'tomato', 'orange', 'gray', 'purple']
    single_split_name = 'single'

    num_delay = 15

    max_x = 500 # unit is ms
    plt.figure()
    fig, ax = plt.subplots(1,2, sharex='col', sharey=False, figsize=(18,6))
    fig.suptitle(f'{title}', fontsize='x-large', y=0.99)
    # fig.text(0.5, 0.03, 'Mean delay of the most recent frame(ms).', ha='center', fontsize='x-large')
    # fig.text(0.08, 0.5, 'AP@0.50', va='center', rotation='vertical', fontsize='x-large')
    # ax30 = ax[0]; 
    ax50 = ax[0]; ax70 = ax[1]
    
    # for single fusion
    method_name = 'Single'
    # eval_file = os.path.join(root_dirs, f'eval_{single_split_name}.yaml')
    eval_file = '/remote-home/share/sizhewei/logs/opv2v_late_fusion/eval_no_noise_single_delay_0.00.yaml'
    single_color = 'red'
    with open(eval_file, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # plt.sca(ax30); plt.plot([0,max_x],[data['ap_30'],data['ap_30']], color=single_color, linestyle='--', label=method_name)
    plt.sca(ax50); plt.plot([100,max_x],[data['ap_50'],data['ap_50']], color=single_color, linestyle='--', label=method_name)
    plt.sca(ax70); plt.plot([100,max_x],[data['ap_70'],data['ap_70']], color=single_color, linestyle='--', label=method_name)

    for split_i, split_name in enumerate(split_list):
        ap_list = []
        delays = []
        if split_name == 'late':
            method_name = 'Late Fusion'
            file_name = 'opv2v_late_fusion'
            eval_name = 'late_delay'
        elif split_name == 'v2vnet':
            method_name = 'V2VNet'
            file_name = 'opv2v_v2vnet_32ch'
            eval_name = 'v2vnet_32ch_ep23'
        elif split_name == 'v2xvit':
            method_name = 'V2X-ViT'
            file_name = 'opv2v_point_pillar_v2xvit_v1_slren'
            eval_name = 'v2xvit'
        elif split_name == 'disconet':
            method_name = 'DiscoNet'
            file_name = 'opv2v_disconet_sren'
            eval_name = 'disconet'
        elif split_name == 'where2comm':
            method_name = 'Where2comm'
            file_name = 'irv2v_where2comm_cobevflow_w_dir_finetune' #'irv2v_where2comm_max_multiscale_resnet'
            eval_name = 'where2comm'
        elif split_name == 'where2comm_syncnet':
            method_name = 'Where2comm + SyncNet'
            file_name = 'irv2v_where2comm_syncnet_regular_100_past_2_test'
            eval_name =  'syncnet_ir_exp_f500' # 'syncnet_e8'
        elif split_name == 'cobevflow':
            method_name = 'CoBEVFlow (ours)'
            file_name = 'irv2v_where2comm_cobevflow_w_dir_finetune' #'opv2v_where2comm_cobevflow_w_dir'
            eval_name = 'cobevflow_ir_exp_f500'
        latest_time_delay = -500.00
        for i in tqdm(np.linspace(0.1, 0.9, 9)):
            # log_file = os.path.join(split_name, f"eval_{note}_%.1f.yaml" % i)
            eval_file = os.path.join(root_dirs, file_name, f"eval_no_noise_{eval_name}_%.2f.yaml" % i)
            
            if not os.path.exists(eval_file):
                print(f'eval file {eval_file} not exist!')
                continue
            with open(eval_file, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            
            unit_time_delay = i*1000 - 500

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
        plt.sca(ax50); plt.plot(delays, ap_50_list, color=color, linewidth='3', label = method_name) # , marker='+'
        plt.sca(ax70); plt.plot(delays, ap_70_list, color=color, linewidth='3', label = method_name) # , marker='+'

    xaxis = np.linspace(-400, 400, 9)
    # ax30.set_title('The Results for AP@0.3'); ax30.grid(True); 
    # ax30.set_xticks(xaxis); \
    #     ax30.legend(loc = 'lower left')
    yaxis = np.linspace(0.5, 0.9, 5)
    # ax50.set_title('The Results for AP@0.5'); 
    ax50.grid(True); 
    ax50.set_xticks(xaxis); ax50.set_yticks(yaxis); 
    ax50.set_xlabel('Expectation time delay of the latest frame (ms)'); ax50.set_ylabel('AP@0.50');
    ax50.legend(loc = 'best', ncol=3)
    
    yaxis = np.linspace(0.4, 0.8, 5)
    # ax70.set_title('The Results for AP@0.7'); 
    ax70.grid(True); 
    ax70.set_xticks(xaxis); ax70.set_yticks(yaxis); 
    ax70.set_xlabel('Expectation time delay of the latest frame (ms)'); ax70.set_ylabel('AP@0.70');
    ax70.legend(loc = 'best', ncol=3)

    save_subfig(fig,ax[0],'./opencood/','ir2v2ap50-irr.pdf')
    save_subfig(fig,ax[1],'./opencood/','irv2vap70-irr.pdf')

    # plt.legend(frameon=False)
    plt.savefig(save_path, dpi=600, format=format)
    print("=== Plt save finished!!! ===")
    # plt.title('标题')
    # plt.show()
