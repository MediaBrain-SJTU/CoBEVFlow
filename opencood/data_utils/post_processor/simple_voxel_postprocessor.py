# Author: Yifan Lu

"""
3D Anchor Generator for Voxel
"""
import numpy as np
import torch
import torch.nn.functional as F
import math

from opencood.data_utils.post_processor.voxel_postprocessor \
    import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils import common_utils


class SimpleVoxelPostprocessor():
    def __init__(self, params, train=True):
        self.params = params
        self.anchor_num = params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7)

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def post_process_stage1(self, stage1_output_dict, anchor_box):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: (project the bounding boxes to ego space). 
            For stage1, we do not project to ego. Latter we project to world coord.
        Step:3 NMS

        Parameters
        ---------
            stage1_output_dict :
                psm: torch.Size([12, 2, 100, 352])
                rm: torch.Size([12, 14, 100, 352])
            anchor_box: torch.Size()

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        psm = stage1_output_dict['psm']
        rm = stage1_output_dict['rm']
        
        prob = psm  # [N, anchor_num , H, W]
        prob = F.sigmoid(prob.permute(0, 2, 3, 1).contiguous())  # [N, H, W, anchor_num]

        # regression map
        reg = rm  # [N, anchor_num * 7, H, W]

        # convert regression map back to bounding box
        batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
        mask = torch.gt(prob, self.params['target_args']['score_threshold'])
        batch_num_box_count = [int(m.sum()) for m in mask]
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)


        boxes3d = torch.masked_select(batch_box3d.view(-1, 7), mask_reg[0]).view(-1, 7) 
        scores = torch.masked_select(prob.view(-1), mask[0])

        # convert output to bounding box
        if len(boxes3d) != 0:
            # save origianl format box. [N, 7]
            pred_box3d_original = boxes3d.detach()
            # (N, 8, 3)
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
            # (N, 8, 3)
            pred_corners_tensor = boxes3d_corner  # box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(pred_corners_tensor)
            # (N, 5)
            pred_box2d_score_tensor = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
            scores = pred_box2d_score_tensor[:, -1]

        else:
             return None, None, None

        # divide boxes to each cav

        cur_idx = 0
        batch_pred_corners3d = [] # [[N1, 8, 3], [N2, 8, 3], ...]
        batch_pred_boxes3d = [] # [[N1, 7], [N2, 7], ...]
        batch_scores = []
        for n in batch_num_box_count:
            cur_corners = pred_corners_tensor[cur_idx: cur_idx+n]
            cur_boxes = pred_box3d_original[cur_idx: cur_idx+n]
            cur_scores = scores[cur_idx:cur_idx+n]
            # nms
            keep_index = box_utils.nms_rotated(cur_corners,
                                               cur_scores,
                                               self.params['nms_thresh']
                                               )
            batch_pred_corners3d.append(cur_corners[keep_index])
            batch_pred_boxes3d.append(cur_boxes[keep_index])
            batch_scores.append(cur_scores[keep_index])
            cur_idx += n

        return batch_pred_corners3d, batch_pred_boxes3d, batch_scores



    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d