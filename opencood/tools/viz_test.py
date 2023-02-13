import os
import sys
sys.path.append(os.getcwd())
from opencood.hypes_yaml.yaml_utils import load_yaml
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import x_to_world, x1_to_x2
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
import numpy as np
from matplotlib import pyplot as plt

cav_list = ['1045', '1054']

pcd_path_list = ['/GPFS/public/OPV2V_Irregular_V2/test/2021_08_18_19_48_05/%s/248.000.pcd' % cav for cav in cav_list]
yaml_path_list = ['/GPFS/public/OPV2V_Irregular_V2/dataset_irregular_v2/test/2021_08_18_19_48_05/%s/248.000.yaml'  % cav for cav in cav_list]

lidar_pose = []
pcd_list = []
for i in range(2):
    pcd_file = pcd_path_list[i]
    lidar = pcd_utils.pcd_to_np(pcd_file)
    pcd_list.append(lidar)

    yaml_file = yaml_path_list[i]
    params = load_yaml(yaml_file)
    tmp_ego_pose = np.array(params['true_ego_pos'])
    tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
    lidar_pose.append(tmp_ego_pose)

other_to_ego_T = x1_to_x2(lidar_pose[1], lidar_pose[0]) # 4,4
ego_pcd = pcd_list[0]
other_pcd = pcd_list[1]
other_pcd[:, -1] = 1 # [N, 4]
other_to_ego_pcd = (other_to_ego_T @ other_pcd.T).T

pcd_np = np.concatenate([ego_pcd, other_to_ego_pcd], 0)

print(pcd_np.shape)

pc_range = [-140.8, -40, -3, 140.8, 40, 1]
pc_range = [int(i) for i in pc_range]

canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True
                                          ) 
canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points

plt.axis("off")

plt.imshow(canvas.canvas)

plt.tight_layout()
plt.savefig('./simple_viz.jpg', transparent=False, dpi=400)
plt.clf()