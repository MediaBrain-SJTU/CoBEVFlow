# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Point pillar loss for time compensation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarTcLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarTcLoss, self).__init__()
        self.loss_dict = {}

        self.backbone_fix = False
        self.use_dir = False
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix = True
            self.flow_loss = nn.SmoothL1Loss(reduction='none')
            self.state_loss = nn.BCEWithLogitsLoss()
        else:
            self.reg_loss_func = WeightedSmoothL1Loss()
            self.alpha = 0.25
            self.gamma = 2.0
            self.cls_weight = args['cls_weight']
            self.reg_coe = args['reg']

            if 'dir_args' in args.keys():
                self.use_dir = True
                self.dir_weight = args['dir_args']['dir_weight']
                self.dir_offset = args['dir_args']['args']['dir_offset']
                self.num_bins = args['dir_args']['args']['num_bins']
                anchor_yaw = np.deg2rad(np.array(args['dir_args']['anchor_yaw']))  # for direction classification
                self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
                self.anchor_num = self.anchor_yaw_map.shape[1]
            else:
                self.use_dir =False

    def delete_ego_label(self, target_dict, record_len):
        '''
        将 target_dict 中每一项的ego部分去掉 

        Parameters: 
        -----------
        target_dict: dict
            {
                'pos_equal_one': # [\sum_{B} N* , H, W, 2]
                'neg_equal_one': # [\sum_{B} N* , H, W, 2]
                'targets':       # [\sum_{B} N* , H, W, 14]
            }
        record_len: list
            len = B, 存放每个batch有多少辆车

        Returns:
        --------
        updated_dict: dict
            {
                'pos_equal_one': # [\sum_{B} n* , H, W, 2]   n* = N* - 1
                'neg_equal_one': # [\sum_{B} n* , H, W, 2]
                'targets':       # [\sum_{B} n* , H, W, 14]
            }
        '''
        updated_dict = {}

        split_loc = torch.cumsum(record_len, dim=0)
        for key, values in target_dict.items():
            deleted_label = []
            batch_list = torch.tensor_split(values, split_loc[:-1].cpu())
            for unit_content in batch_list:
                deleted_label.append(unit_content[1:])
            updated_dict[key] = torch.cat(deleted_label, dim=0)
        return updated_dict


    def forward(self, output_dict, target_dict, mode=None, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if self.backbone_fix:
            total_loss = 0
            ###### flow loss ######
            if f'flow_preds{suffix}' in output_dict and output_dict[f'flow_preds{suffix}'].shape[0] != 0:
                h, w = output_dict[f'flow_preds{suffix}'].shape[-2], output_dict[f'flow_preds{suffix}'].shape[-1]
                output_dict[f'flow_preds{suffix}'] = output_dict[f'flow_preds{suffix}'].reshape(1, -1, h, w)
                target_dict[f'flow_gt{suffix}'] = target_dict[f'flow_gt{suffix}'].reshape(1, -1, h, w)
                flow_loss = self.flow_loss(output_dict[f'flow_preds{suffix}'],
                                        target_dict[f'flow_gt{suffix}'])
                valid_flow_mask = ((torch.abs(target_dict[f'flow_gt{suffix}'].reshape(-1, 2, h, w)).max(dim=1)[0] > 0.49) * 1.0).unsqueeze(1)
                flow_loss = (flow_loss * valid_flow_mask).sum() / (valid_flow_mask.sum() + 1e-6)
                # flow_loss *= self.flow_weight

                state_loss = self.state_loss(output_dict[f'state_preds{suffix}'],
                                            valid_flow_mask)

                total_loss = flow_loss + state_loss
                self.loss_dict.update({'total_loss': total_loss.item(),
                                    'flow_loss': flow_loss.item(),
                                    'state_loss': state_loss.item()})
        else:
            if mode=='curr':
                rm = output_dict['rm_curr']
                psm = output_dict['psm_curr']
            elif mode=='latency':
                rm = output_dict['rm_latency']
                psm = output_dict['psm_latency']
            elif mode == 'single':
                psm = output_dict['psm_nonego_single']              # [\sum_{B}n, 2, H, W]
                rm = output_dict['rm_nonego_single']                # [\sum_{B}n, 14, H, W]
                target_dict = self.delete_ego_label(target_dict, output_dict['record_len'])
            else:
                rm = output_dict['rm']  # [B, 14, 50, 176]
                psm = output_dict['psm'] # [B, 2, 50, 176]
                if self.use_dir:
                    dm = output_dict["dm"] # [B, 4, 50, 176]

            # rename variable 
            if f'psm{suffix}' in output_dict:
                psm = output_dict[f'psm{suffix}']
            if f'rm{suffix}' in output_dict:
                rm = output_dict[f'rm{suffix}']
            if self.use_dir:
                if f'dm{suffix}' in output_dict:
                    dm = output_dict[f'dm{suffix}']
            
            targets = target_dict['targets']                # B, H, W, 14
            box_cls_labels = target_dict['pos_equal_one']   # B, H, W, 2

            cls_preds = psm.permute(0, 2, 3, 1).contiguous() # N, C, H, W -> N, H, W, C
            box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()     # B, HxWx2

            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), 2,
                dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(psm.shape[0], -1, 1)
            one_hot_targets = one_hot_targets[..., 1:]

            cls_loss_src = self.cls_loss_func(cls_preds,
                                            one_hot_targets,
                                            weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / psm.shape[0]
            conf_loss = cls_loss * self.cls_weight

            # regression
            rm = rm.permute(0, 2, 3, 1).contiguous()
            rm = rm.view(rm.size(0), -1, 7)
            targets = targets.view(targets.size(0), -1, 7)
            box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                    targets)
            loc_loss_src =\
                self.reg_loss_func(box_preds_sin,
                                reg_targets_sin,
                                weights=reg_weights)
            reg_loss = loc_loss_src.sum() / rm.shape[0]
            reg_loss *= self.reg_coe

            ######## direction ##########
            if self.use_dir:
                dir_targets = self.get_direction_target(targets)
                N =  dm.shape[0]
                dir_logits = dm.permute(0, 2, 3, 1).contiguous().view(N, -1, 2) # [N, H*W*#anchor, 2]

                dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
                dir_loss = dir_loss.view(dir_logits.shape[:2]) * reg_weights # [N, H*W*anchor_num]
                dir_loss = dir_loss.sum() * self.dir_weight / N

            total_loss = reg_loss + conf_loss
            
            self.loss_dict.update({'total_loss': total_loss,
                                'reg_loss': reg_loss,
                                'conf_loss': conf_loss})

            if self.use_dir:
                total_loss += dir_loss
                self.loss_dict.update({'dir_loss': dir_loss})

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        # (1, 2, 1)
        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - self.dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / self.num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=self.num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, self.num_bins)
        return dir_cls_targets



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        flow_loss = self.loss_dict.get('flow_loss', 0)
        state_loss = self.loss_dict.get('state_loss', 0)
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        conf_loss = self.loss_dict.get('conf_loss', 0)
            
        print_msg = ("[epoch %d][%d/%d]%s, || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f || Flow Loss: %.4f || State Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len, suffix, 
                        total_loss, conf_loss, reg_loss, flow_loss, state_loss))
        
        if self.use_dir:
            dir_loss = self.loss_dict['dir_loss']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        print(print_msg)

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, conf_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Flow_recon_loss'+suffix, flow_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('State_loss'+suffix, state_loss,
                            epoch*batch_len + batch_id)
                            
            if self.use_dir:
                writer.add_scalar('dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
