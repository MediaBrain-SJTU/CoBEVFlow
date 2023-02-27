import torch
from matplotlib import pyplot as plt
import numpy as np

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

from opencood.utils import box_utils
import os

def visualize(pred_box_tensor, comp_box_tensor, gt_tensor, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True, vis_comp_box=True, left_hand=False, uncertainty=None):
    """
    Visualize the 
    1. [prediction], 
    2. [compensation]
    3. ground truth 
    4. with point cloud 
    together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (n, 8, 3) prediction.

    comp_box_tensor : torch.Tensor
        (n, 8, 3) compensation bbx

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """

    pc_range = [int(i) for i in pc_range]
    pcd_np = pcd.cpu().numpy()

    if vis_pred_box:
        pred_box_np = pred_box_tensor.cpu().numpy()
        pred_name = ['pred'] * pred_box_np.shape[0]
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6**2 + 3.9**2
            
            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) 
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:,:2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                for i in range(uncertainty_np.shape[0])]                    

    if vis_comp_box:
        comp_box_np = comp_box_tensor.cpu().numpy()
        comp_name = ['comp'] * comp_box_np.shape[0]

    if vis_gt_box:
        gt_box_np = gt_tensor.cpu().numpy()
        gt_name = ['gt'] * gt_box_np.shape[0]

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                        canvas_x_range=(pc_range[0], pc_range[3]), 
                                        canvas_y_range=(pc_range[1], pc_range[4]),
                                        left_hand=left_hand
                                        ) 

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
        if vis_comp_box:
            canvas.draw_boxes(comp_box_tensor, colors=(0,0,255), texts=comp_name)

    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
        
    else:
        raise(f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400)
    plt.clf()


def viz_compensation_latefusion_flow(dataset, batch_data, box_results, file_name='debug_viz', save_notes='', vis_comp_box = True, batch_id=0):
    k = 2
    vis_save_file = f'/DB/data/sizhewei/logs/latefusion_flow/{file_name}/'
    if not os.path.exists(vis_save_file):
        os.mkdir(vis_save_file)
    gt_range = torch.tensor([-140.8, -40, -3, 140.8, 40, 1])
    for cav_id, cav_content in box_results.items():
        gt_box_tensor = dataset.generate_gt_bbx_cav_curr(batch_data[cav_id])
        for i in range(k):
            pred_box_tensor = cav_content[i]['pred_box_3dcorner_tensor']
            projected_box_tensor = box_utils.project_box3d(pred_box_tensor, batch_data[cav_id]['trans_mat_cavpast0_2_cavcurr_torch'])
            vis_save_path = vis_save_file + f'b_{batch_id}_{cav_id}_past_{i}_{save_notes}.png'
            if vis_comp_box:
                comp_box_tensor = cav_content['comp']['pred_box_3dcorner_tensor']
                comp_box_tensor = box_utils.project_box3d(comp_box_tensor, batch_data[cav_id]['trans_mat_cavpast0_2_cavcurr_torch']).cpu()
            else:
                comp_box_tensor = None
            visualize(projected_box_tensor.cpu(),
                    comp_box_tensor,
                    gt_box_tensor.cpu(),
                    batch_data[cav_id]['origin_lidar'][0],
                    gt_range,
                    vis_save_path,
                    method='bev',
                    left_hand=True,
                    uncertainty=None, 
                    vis_comp_box=vis_comp_box)
            