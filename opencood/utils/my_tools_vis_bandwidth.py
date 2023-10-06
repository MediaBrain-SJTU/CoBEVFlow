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
import math
import matplotlib.pyplot as plt

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import pcd_to_np

# from brokenaxes import brokenaxes
from scipy import interpolate

# 用于单独保存子图的函数
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path+fig_name, bbox_inches=extent, dpi=600, format='png')


def export_legend(ax, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=3,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=600, bbox_inches=bbox)

if __name__ == "__main__":
    # to_npy_and_json(split_name='validate')
    # create_small_dataset(split_name='train')

    root_dirs = "/remote-home/share/sizhewei/logs/bandwidth_curve"
    note = 'major_bandwidth'
    format = 'jpg'
    save_path = f"./opencood/result_{note}.{format}"
    title = "Average Precision curves of different methods on the IRV2V dataset at different average time intervals."
    plt.style.use('ggplot')
    split_list = ['single', 'v2vnet', 'v2xvit', 'disconet']
    marker_list = []
    split_list_sparse = ['where2comm', 'syncnet', 'cobevflow']
    colors = ['gray', '#DEC68B', '#BEB8DC', '#82B0D2']
    # colors = ['#999999', '#DEC68B', '#BEB8DC', '#82B0D2', '#8ECFC9']  #'#FA7F6F'] #'#E7DAD2'
    colors_sparse = ['#8ECFC9', '#FFBE7A', 'red']


    max_x = 17.5 # unit is ms
    linewidth= '7'
    fontsize = 22

    plt.figure()
    fig, ax = plt.subplots(1,2, sharex='col', sharey=False, figsize=(16,7))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.1)

    ax50 = ax[0]; ax70 = ax[1]

    for split_i, split_name in enumerate(split_list):
        if split_name == 'single':
            method_name = 'Single'
            eval_name = 'single_delay'
        elif split_name == 'late':
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
            file_name = 'opv2v_disconet_new_ylu'
            eval_name = 'disconet'
        
        color = colors[split_i]
        eval_file = os.path.join(root_dirs,f"eval_no_noise_{eval_name}_0.30.yaml")
            
        if not os.path.exists(eval_file):
            print(f'eval file {eval_file} not exist!')
            continue
        with open(eval_file, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        bandwith = math.log(200*704, 2)
        if split_name == 'single':
            bandwith = 0
        plt.sca(ax50); plt.scatter(bandwith, data['ap_50'], color=color, marker='o', s=400, label = method_name)
        plt.sca(ax70); plt.scatter(bandwith, data['ap_70'], color=color, marker='o', s=400, label = method_name) 


    for split_i, split_name in enumerate(split_list_sparse):
        ap_list = []
        bw = []
        if split_name == 'where2comm':
            method_name = 'Where2Comm'
            file_name = 'opv2v_late_fusion'
            eval_name = 'where2comm'
        elif split_name == 'syncnet':
            method_name = 'Where2comm + SyncNet'
            eval_name = 'syncnet'
        elif split_name == 'cobevflow':
            method_name = 'CoBEVFlow (ours)'
            file_name = 'opv2v_v2vnet_32ch'
            eval_name = 'cobevflow_regular'

        for i in [5, 10, 15, 20, 25, 40, 3520]:
            eval_file = os.path.join(root_dirs,f"eval_no_noise_{eval_name}_0.30_noise_0_0_0_0_roi_{i}.yaml")
            if i==0:
                eval_file = '/remote-home/share/sizhewei/logs/bandwidth_curve/eval_no_noise_where2comm_no_fusion_0.30_noise_0_0_0_0_roi_0.yaml'
            if not os.path.exists(eval_file):
                print(f'eval file {eval_file} not exist!')
                continue
            with open(eval_file, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            
           
            if i==0:
                bw.append(0)
            else:
                bw.append(math.log(i*40, 2))

            tmp_aps = []
            tmp_aps.append(data['ap_30'])
            tmp_aps.append(data['ap_50'])
            tmp_aps.append(data['ap_70'])

            ap_list.append(tmp_aps)

        color = colors_sparse[split_i]
        ap_list_np = np.array(ap_list)
        ap_list_np = np.transpose(ap_list_np)

        ap_30_list = list(ap_list_np[0])
        ap_50_list = list(ap_list_np[1])
        ap_70_list = list(ap_list_np[2])

        # bw_new = np.linspace(0, 17.103, 170)
        
        # bspline = interpolate.make_interp_spline(bw, ap_50_list, k=4)
        # ap_50_list_new = bspline(bw_new)

        # bspline = interpolate.make_interp_spline(bw, ap_70_list, )
        # ap_70_list_new = bspline(bw_new)


        # plt.sca(ax50); plt.plot(bw_new, ap_50_list_new, color=color, linewidth='3', label = method_name, marker='+') # , marker='+'
        # plt.sca(ax70); plt.plot(bw_new, ap_70_list_new, color=color, linewidth='3', label = method_name, marker='+')

        plt.sca(ax50); plt.plot(bw, ap_50_list, color=color, linewidth=linewidth, label = method_name)#, marker='+') # , marker='+'
        plt.sca(ax70); plt.plot(bw, ap_70_list, color=color, linewidth=linewidth, label = method_name)#, marker='+')

    xaxis = np.linspace(0, 15, 5)
    # ax30.set_title('The Results for AP@0.3'); ax30.grid(True); 
    # ax30.set_xticks(xaxis); \
    #     ax30.legend(loc = 'lower left')
    yaxis = np.linspace(0.4, 0.9, 6)
    # font = "Gill Sans MT Condensed"
    # ax50.set_title('The Results for AP@0.5'); 
    ax50.tick_params(axis='both', which='major', labelsize=16, color='gray')
    ax50.grid(True); 
    ax50.set_xticks(xaxis); ax50.set_yticks(yaxis);
    ax50.set_xlabel('Communication volume (log2)', fontsize=fontsize, color='black'); ax50.set_ylabel('AP@0.50', fontsize=fontsize, color='black');
    # ax50.legend(loc = 'lower left')
    
    yaxis = np.linspace(0.3, 0.8, 6)
    # ax70.set_title('The Results for AP@0.7'); 
    ax70.grid(True); 
    ax70.tick_params(axis='both', which='major', labelsize=16)
    ax70.set_xticks(xaxis); ax70.set_yticks(yaxis); 
    ax70.set_xlabel('Communication volume (log2)', fontsize=fontsize, color='black'); ax70.set_ylabel('AP@0.70', fontsize=fontsize, color='black');
    # ax70.legendloc = 'lower left')

    save_subfig(fig,ax[0],'./opencood/','bandwidth-ap50.png')
    save_subfig(fig,ax[1],'./opencood/','bandwidth-ap70.png')

    # plt.legend(frameon=False)
    plt.savefig(save_path, dpi=600, format=format)
    print("=== Plt save finished!!! ===")

    export_legend(ax70, '/root/percp/OpenCOOD/opencood/legend-bandwidth.pdf')
    # # then create a new image
    # # adjust the figure size as necessary
    # figsize = (7, 7)
    # fig_leg = plt.figure(figsize=figsize)
    # ax_leg = fig_leg.add_subplot(111)
    # # add the legend from the previous axes
    # # ax70.legend(ncol=7)
    # ax_leg.legend(*ax70.get_legend_handles_labels(), loc='center')
    # # hide the axes frame and the x/y labels
    # ax_leg.axis('off')
    # fig_leg.savefig('/root/percp/OpenCOOD/opencood/legend.jpg')

    # plt.title('标题')
    # plt.show()
