import torch
import torch.nn.functional as F 
from matplotlib import pyplot as plt 

import numpy as np
import cv2
import copy
from functools import partial
import matplotlib


pc_list = [-140.8, -40, -3, 140.8, 40, 1]
pc_range = list((np.array(pc_list)*1.25).astype(np.int64))

class Canvas_BEV_heading_right(object):
    def __init__(self,
                 canvas_feature, 
                 canvas_shape=(800, 2800),  # 100, 352
                 canvas_x_range=(-140, 140),  # -176, 176
                 canvas_y_range=(-40, 40),      # -50, 50
                 canvas_bg_color=(0, 0, 0),
                 left_hand=True):
        """
        Args:
            canvas_shape (Tuple[int]): Shape of BEV Canvas image. First element
                corresponds to Y range, the second element to X range.
            canvas_x_range (Tuple[int]): Range of X-coords to visualize. X is
                horizontal: negative ~ positive is left ~ right.
            canvas_y_range (Tuple[int]): Range of Y-coords to visualize. Y is
                vertcal: negative ~ positive is top ~ down.
            canvas_bg_color (Tuple[int]): RGB (0 ~ 255) of Canvas background
                color.
            left_hand: (bool), whether the point cloud is left-hand coordinate
                V2X-Sim is right hand, and OPV2V is left hand.

            Different from Canvas_BEV, the vehicle is heading right.
            Naturally this code is designed for left hand coordinate

        """
        
        # Sanity check ratios
        if ((canvas_shape[1] / canvas_shape[0]) != 
            ((canvas_x_range[0] - canvas_x_range[1]) / 
             (canvas_y_range[0] - canvas_y_range[1]))):

            print("Not an error, but the x & y ranges are not "\
                  "proportional to canvas height & width.")
        
        self.canvas_feature = canvas_feature
        self.canvas_shape = canvas_shape
        self.canvas_x_range = canvas_x_range
        self.canvas_y_range = canvas_y_range
        self.canvas_bg_color = canvas_bg_color
        self.left_hand = left_hand
        
        self.clear_canvas()
    
    def get_canvas(self):
        return self.canvas

    def clear_canvas(self):
        feature_norm = (self.canvas_feature - np.min(self.canvas_feature)) /\
             (np.max(self.canvas_feature) - np.min(self.canvas_feature))
        img = (feature_norm*255).astype(np.uint8)
        
        resize_img = cv2.resize(img, (img.shape[1]*10, img.shape[0]*10), interpolation=cv2.INTER_LINEAR)

        self.canvas = resize_img
        # self.canvas = np.zeros((*self.canvas_shape, 3), dtype=np.uint8)
        # self.canvas[..., :] = self.canvas_bg_color
        # self.canvas = np.zeros((*self.canvas_shape, 1))
        # self.canvas[..., :] = self.canvas_feature[:,:,np.newaxis]

    def get_canvas_coords(self, xy):
        """
        Args:
            xy (ndarray): (N, 2+) array of coordinates. Additional columns
                beyond the first two are ignored.
        
        Returns:
            canvas_xy (ndarray): (N, 2) array of xy scaled into canvas 
                coordinates. Invalid locations of canvas_xy are clipped into 
                range. "x" is dim0, "y" is dim1 of canvas.
            valid_mask (ndarray): (N,) boolean mask indicating which of 
                canvas_xy fits into canvas.
        """
        xy = np.copy(xy) # prevent in-place modifications

        x = xy[:, 0]
        y = xy[:, 1]

        if not self.left_hand:
            y = -y

        # Get valid mask
        valid_mask = ((x > self.canvas_x_range[0]) & 
                      (x < self.canvas_x_range[1]) &
                      (y > self.canvas_y_range[0]) & 
                      (y < self.canvas_y_range[1]))

        # Rescale points
        # They are exactly lidar point coordinate
        x = ((x - self.canvas_x_range[0]) / 
             (self.canvas_x_range[1] - self.canvas_x_range[0]))
        x = x * self.canvas_shape[1]
        x = np.clip(np.around(x), 0, 
                    self.canvas_shape[1] - 1).astype(np.int32) # [0,2800-1]
                    
        y = ((y - self.canvas_y_range[0]) / 
             (self.canvas_y_range[1] - self.canvas_y_range[0]))
        y = y * self.canvas_shape[0]
        y = np.clip(np.around(y), 0, 
                    self.canvas_shape[0] - 1).astype(np.int32) # [0,800-1]
        
        # x and y are exactly image coordinate
        # ------------> x
        # |
        # |
        # |
        # y

        canvas_xy = np.stack([x, y], axis=1)

        return canvas_xy, valid_mask
                                      
    def draw_canvas_points(self, 
                           canvas_xy,
                           radius=-1,
                           colors=None,
                           colors_operand=None):
        """
        Draws canvas_xy onto self.canvas.

        Args:
            canvas_xy (ndarray): (N, 2) array of *valid* canvas coordinates.
                
            radius (Int): 
                -1: Each point is visualized as a single pixel.
                r: Each point is visualized as a circle with radius r.
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
                String: Such as "Spectral", uses a matplotlib cmap, with the
                    operand (the value cmap is called on for each point) being 
                    colors_operand. If colors_operand is None, uses normalized
                    distance from (0, 0) of XY point coords.
            colors_operand (ndarray | None): (N,) array of values cooresponding
                to canvas_xy, to be used only if colors is a cmap.
        """
        if len(canvas_xy) == 0:
            return 
            
        if colors is None:
            colors = np.full(
                (len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., :] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        elif isinstance(colors, str):
            colors = matplotlib.cm.get_cmap(colors)
            if colors_operand is None:
                # Get distances from (0, 0) (albeit potentially clipped)
                origin_center = self.get_canvas_coords(np.zeros((1, 2)))[0][0]
                colors_operand = np.sqrt(
                    ((canvas_xy - origin_center) ** 2).sum(axis=1))
                    
            # Normalize 0 ~ 1 for cmap
            colors_operand = colors_operand - colors_operand.min()
            colors_operand = colors_operand / colors_operand.max()
        
            # Get cmap colors - note that cmap returns (*input_shape, 4), with
            # colors scaled 0 ~ 1
            colors = (colors(colors_operand)[:, :3] * 255).astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))

        # Here the order is different from Canvas_BEV
        if radius == -1:
            self.canvas[canvas_xy[:, 1], canvas_xy[:, 0], :] = colors
        else:
            for color, (x, y) in zip(colors.tolist(), canvas_xy.tolist()):
                self.canvas = cv2.circle(self.canvas, (x, y), radius, color, 
                                         -1, lineType=cv2.LINE_AA)

    def draw_boxes(self,
                   boxes,
                   colors=None,
                   texts=None,
                   box_line_thickness=2,
                   box_text_size=0.5,
                   text_corner=0):
        """
        Draws a set of boxes onto the canvas.
        Args:
            boxes (ndarray): [N, 8, 3] corner 3d
                
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
            texts (List[String]): Length N; text to write next to boxes.
            box_line_thickness (int): cv2 line/text thickness
            box_text_size (float): cv2 putText size
            text_corner (int): 0 ~ 3. Which corner of 3D box to write text at.
        """
        # Setup colors
        if colors is None:
            colors = np.full((len(boxes), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(boxes), 3), dtype=np.uint8)
            colors_tmp[..., :len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(boxes)
            colors = colors.astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))

        boxes = np.copy(boxes) # prevent in-place modifications
        

        # Translate BEV 4 corners , [N, 4, 2]
        #     4 -------- 5
        #    /|         /|
        #   7 -------- 6 .
        #   | |        | |
        #   . 0 -------- 1
        #   |/         |/
        #   3 -------- 2
        bev_corners = boxes[:,:4,:2]     # [N, 4, 2]

        ## Transform BEV 4 corners to canvas coords
        bev_corners_canvas, valid_mask = \
            self.get_canvas_coords(bev_corners.reshape(-1, 2))  # [N, 2]
        bev_corners_canvas = bev_corners_canvas.reshape(*bev_corners.shape)  # [N, 4, 2]
        valid_mask = valid_mask.reshape(*bev_corners.shape[:-1]) 

        # At least 1 corner in canvas to draw.
        valid_mask = valid_mask.sum(axis=1) > 0
        bev_corners_canvas = bev_corners_canvas[valid_mask]
        if texts is not None:
            texts = np.array(texts)[valid_mask]

        ## Draw onto canvas
        # Draw the outer boundaries
        idx_draw_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, (color, curr_box_corners) in enumerate(
                zip(colors.tolist(), bev_corners_canvas)):
                
            curr_box_corners = curr_box_corners.astype(np.int32)
            for start, end in idx_draw_pairs:
                # Notice Difference Here
                self.canvas = cv2.line(self.canvas,
                                       tuple(curr_box_corners[start]\
                                        .tolist()),
                                       tuple(curr_box_corners[end]\
                                        .tolist()),
                                       color=color,
                                       thickness=box_line_thickness)
            if texts is not None:
                self.canvas = cv2.putText(self.canvas,
                                          str(texts[i]),
                                          tuple(curr_box_corners[text_corner]\
                                            .tolist()),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          box_text_size,
                                          color=color,
                                          thickness=box_line_thickness)

def viz_on_canvas(feature, bbox_list):
    """ 
    feature: torch.Tensor, (C, H, W) 
    bbox_list: torch.Tensor, (n, 2)
    """
    canvas = Canvas_BEV_heading_right(torch.max(feature, dim=0)[0].cpu().numpy(),
                                    canvas_shape=(int((pc_range[4]-pc_range[1]))*10, \
                                        int((pc_range[3]-pc_range[0]))*10),
                                    canvas_x_range=(pc_range[0], pc_range[3]), 
                                    canvas_y_range=(pc_range[1], pc_range[4]),
                                    left_hand=True) 
    canvas.draw_boxes((bbox_list).cpu().numpy(), colors=(255,0,0))
    return canvas

def feature_warp(feature, bbox_list, flow, align_corners=False):
    """
    Parameters
    -----------
    feature: [C, H, W]
    bbox_list: [num_cav, 4, 3] at cav coodinate system
    flow:[num_cav, 2] at cav coodinate system
        bbox_list & flow : x and y are exactly image coordinate
        ------------> x
        |
        |
        |
        y
    
    Returns
    -------
    updated_feature: feature after being warped by flow, [C, H, W]
    """
    # flow = torch.tensor([70, 0]).unsqueeze(0).to(feature)

    if flow.shape[0] == 0 : 
        return feature

    # only use x and y
    bbox_list = bbox_list[:, :, :2]

    # scale meters to voxel, feature_length / lidar_range_length = 1.25
    flow = flow * 1.25 
    bbox_list = bbox_list * 1.25

    # store two parts of bbx: 1. original bbx, 2. 
    viz_bbx_list = bbox_list
    fig, ax = plt.subplots(4, 1, figsize=(5,11))
    
    ######## viz-0: original feature, original bbx
    canvas_ori = viz_on_canvas(feature, bbox_list)
    plt.sca(ax[0])
    # plt.axis("off")
    plt.imshow(canvas_ori.canvas)
    ##########

    C, H, W = feature.size()
    num_cav = bbox_list.shape[0]
    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

    '''
    create affine matrix:
    ------------
    1  0  -2*t_y/W
    0  1  -2*t_x/H
    0  0    1 
    ------------
    '''
    flow_clone = flow.detach().clone()

    affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
    flow_clone = -2 * flow_clone / torch.tensor([feature.shape[2], feature.shape[1]]).to(feature)
    # flow_clone = flow_clone[:, [1, 0]]
    affine_matrices[:, :2, 2] = flow_clone 
    
    cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
    print("cav_t_mat", cav_t_mat)

    cav_warp_mat = F.affine_grid(cav_t_mat,
                        [num_cav, C, H, W],
                        align_corners=align_corners).to(feature) # .to() 统一数据格式 float32
    
    ######### viz-1: original feature, original bbx and flowed bbx
    flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
    viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
    canvas_hidden = viz_on_canvas(feature, viz_bbx_list)
    plt.sca(ax[1])
    # plt.axis("off") 
    plt.imshow(canvas_hidden.canvas)
    ##########

    x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
    x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
    y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
    y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
    x_min_fid = (x_min + 176).to(torch.int)
    x_max_fid = (x_max + 176).to(torch.int)
    y_min_fid = (y_min + 50).to(torch.int)
    y_max_fid = (y_max + 50).to(torch.int)

    for cav in range(num_cav):
        basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

    final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
    
    ####### viz-2: warped feature, flowed box and warped 
    p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
    p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
    p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
    p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
    warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
    viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
    canvas_new = viz_on_canvas(final_feature, viz_bbx_list)
    plt.sca(ax[2]) 
    # plt.axis("off") 
    plt.imshow(canvas_new.canvas)
    ############## 

    ####### viz-3: mask area out of warped bbx
    partial_feature_one = torch.zeros_like(feature)  # C, H, W
    for cav in range(num_cav):
        partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
    masked_final_feature = partial_feature_one * final_feature
    canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list)
    plt.sca(ax[3]) 
    # plt.axis("off") 
    plt.imshow(canvas_hidden.canvas)
    ##############

    plt.tight_layout()
    plt.savefig('result_canvas.jpg', transparent=False, dpi=400)
    plt.clf()

    fig, axes = plt.subplots(2, 1, figsize=(4, 4))
    major_ticks_x = np.linspace(0,350,8)
    minor_ticks_x = np.linspace(0,350,15)
    major_ticks_y = np.linspace(0,100,3)
    minor_ticks_y = np.linspace(0,100,5)
    for i, ax in enumerate(axes):
        plt.sca(ax); #plt.axis("off")
        ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which='major', color='w', linewidth=0.4)
        ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
        if i==0:
            plt.imshow(torch.max(feature, dim=0)[0].cpu())
        else:
            plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
    plt.tight_layout()
    plt.savefig('result_features.jpg', transparent=False, dpi=400)
    plt.clf()

    return final_feature

if __name__ == '__main__':
    viz_save_path = '/DB/rhome/sizhewei/percp/OpenCOOD/opencood/viz_out/debug_4_feature_flow'
    feature = torch.load(viz_save_path+'/feature.pt')
    bbox_list = torch.load(viz_save_path+'/bbx_list.pt')
    flow = torch.load(viz_save_path+'/flow.pt')
    # feature = torch.zeros(1, 10, 15).to(torch.float)
    # feature[0, 2, 2] = 1 ; feature[0, 2, 3]=1
    # bbox_list = torch.tensor([[1, 1],[1, 4],[3, 4],[3, 1]]).unsqueeze(0)
    # flow = torch.tensor([[6, 7]])
    print(feature.shape)
    print(bbox_list.shape)
    print(flow.shape)


    # flow = flow*100

    bbox_list = bbox_list[0].unsqueeze(0)
    # flow = flow[0].unsqueeze(0)
    # bbox_list = bbox_list[3:6]
    # flow = flow[3:6]
    print(f'flow = {flow}')
    final_feature = feature_warp(feature, bbox_list, flow)
