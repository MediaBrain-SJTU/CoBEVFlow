"""
Draw the feature flow with arrow
"""

import torch
import numpy as np
import os
import sys
import glob
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


# 箭头长度
arrow_len = 0.5
# 箭头缩放比例
scale = 20
# 色轮colormap，用于表示箭头的方向
cmap = plt.cm.hsv


def draw_feature_flow(file_idx, pt_file, save_path):
    """
    box_dict : {
        'single_detection_bbx': # dict, [0, 1, ... , N-1], 
            [i]: dict:{
                [0]/[1]/[2]: past 3 frames detection results:{  past 3 frames detection results:{ # 都是在[0]的各自坐标系下的检测结果。
                    pred_box_3dcorner_tensor
                    pred_box_center_tensor
                    scores
                }
            }
        }
        'lidar_pose_0': 过去第一帧 所有车的pose [N, 6]
        'lidar_pose_current': 当前帧 所有车的pose [N, 6]
        'matched_idx_list': matched_idx_list, # len=N-1, 表示每个non-ego的过去两帧的匹配id, each of which is [N_obj, 2], 比如 ['matched_idx_list'][0] shape(22,2) 表示过去第一帧的22个框与第二帧的22个框的索引的匹配情况
        'compensated_results_list': # len=N-1, 每个non-ego补偿出来的box, each of which is [N_obj, 4, 3], 注意这里用的是4个点, 最后一个维度上是 xyz, z是多余的
        'single_gt_box_tensor': # list, len=N, 表示ego与non-ego每辆车在current时刻的检测框结果, 例如box_dict['single_gt_box_tensor'][1]大小为[N_obj, 8, 3] 表示第二辆车在current时刻的检测框结果
        'single_lidar': # list, len=N, 表示ego与non-ego每辆车在current时刻的lidar np
        'gt_range': # [-140.8, -40, -3, 140.8, 40, 1], 表示lidar的范围
        'single_updated_feature': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的更新后的feature
        'single_original_feature': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的原始feature
        'single_flow_map': tensor, [N, H, W, 2], 表示ego与non-ego每辆车在past-0时刻的flow map
        'single_reserved_mask': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的reserved mask
    }
    """

    data = torch.load(pt_file)

    agent_num = len(data["single_detection_bbx"])
    left_hand = True
    pc_range = data['gt_range']

    single_flow_map = data['single_flow_map']
    single_flow_map_np = data['single_flow_map'].cpu().numpy()

    H, W = single_flow_map.shape[1], single_flow_map.shape[2]
    identity_flow_np = torch.nn.functional.affine_grid(torch.eye(2, 3).unsqueeze(0), torch.Size((1, 1, H, W))).numpy()

    delta_flow_map_np =  identity_flow_np - single_flow_map_np
    delta_flow_map_np[..., 0] *= 140.8 / 2
    delta_flow_map_np[..., 1] *= 40 / 2

    delta_flow_map_np_abs = np.abs(delta_flow_map_np)

    for i in range(1, agent_num):
        delta_flow_map_np_single = delta_flow_map_np[i]
        # get object center
        obj_center = data['single_detection_bbx'][i][0]['pred_box_center_tensor'].cpu().numpy()
        obj_center_x = (obj_center[:, 0] - (-140.8)) / 0.4
        obj_center_x = obj_center_x.astype(int)
        obj_center_y = (obj_center[:, 1] - (-40)) / 0.4
        obj_center_y = obj_center_y.astype(int)

        obj = np.vstack([obj_center_y, obj_center_x]).T
        print(obj.shape)
        # 箭头的长度和颜色深度的最大值
        arrow_len = np.sqrt(np.sum(delta_flow_map_np_single**2, axis=2)).max()

        # 绘图
        fig, ax = plt.subplots()
        ax.imshow(np.ones_like(delta_flow_map_np_single[:, :, 0], dtype=np.uint8) * 255, cmap='gray_r')

        for arr in obj:
            # 计算箭头的长度
            x = delta_flow_map_np_single[arr[0], arr[1], 0]
            y = delta_flow_map_np_single[arr[0], arr[1], 1]
            length = np.sqrt(x ** 2 + y ** 2) / 0.4 # normalize by grid size

            # 计算箭头的方向
            angle = np.arctan2(y, x)

            feat_hsv = np.zeros((1, 3))

            feat_hsv[:, 0] = (angle + np.pi) / (2 * np.pi)
            feat_hsv[:, 1] = min(length / arrow_len , 1.0)# * scale
            print('===',  length / arrow_len,  '===')
            feat_hsv[:, 2] = 1

            feat_rgb = hsv_to_rgb(feat_hsv)
            print('rgb', feat_rgb)
            print('cmap', cmap(angle / (2 * np.pi)))

            # 绘制箭头
            # plt.arrow(arr[1], arr[0], length * np.cos(angle), length * np.sin(angle), color=cmap(angle / (2 * np.pi)), width=1,head_width=3, head_length=0.5)
            plt.arrow(arr[1], arr[0], length * np.cos(angle), length * np.sin(angle), color=feat_rgb[0], width=0.6,head_width=2, head_length=0.7)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_path, "{}_flow_map_arror_{}.png".format(file_idx, i)), dpi=500)
        plt.close()

def main():
    data_dir = '/remote-home/share/sizhewei/logs/irv2v_where2comm_cobevflow_w_dir_finetune/vis_cobevflow_reverse_viz_debug_3_3_3_new_0.30/bbx_folder'
    files = glob.glob(os.path.join(data_dir, '*.pt'))
    files.sort()
    save_path = os.path.join(data_dir, 'flow_map_arrow')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(files)):
        draw_feature_flow(i, files[i], save_path)

if __name__ == "__main__":
    main()

