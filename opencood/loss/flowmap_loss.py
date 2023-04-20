# sizhewei @ 2023/04/17
# for flowmap loss
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlowMapLoss(nn.Module):
    def __init__(self, args):
        super(FlowMapLoss, self).__init__()
    def forward(self, output_dict, foo=None):
        self.loss = output_dict['flow_recon_loss']
        return self.loss
    def logging(self, epoch, batch_id, batch_len, writer = None):
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
        total_loss = self.loss
        print_msg = ("[epoch %d][%d/%d], || Loss: %.4f " 
        % (epoch, batch_id + 1, batch_len,total_loss.item()))
        print(print_msg)
        if writer is not None:
            writer.add_scalar('loss', total_loss.item(), epoch * batch_len + batch_id)