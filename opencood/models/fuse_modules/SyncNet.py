import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SyncLSTM(nn.Module):
    def __init__(self, channel_size = 256, h = 32, w = 32, k = 3, TM_Flag = True, compressed_size = 32):
        super(SyncLSTM, self).__init__()
        self.k = k

        self.channel_size = channel_size
        self.compressed_size = compressed_size
        self.time_weight = Time_Modulation(input_channel = 2*self.compressed_size)
        self.lstmcell = MotionLSTM(h,w, self.compressed_size)
        self.init_c = nn.parameter.Parameter(torch.rand(self.compressed_size, h, w))
        self.TM_Flag = TM_Flag

        self.ratio = int(math.sqrt(channel_size / compressed_size))
        self.conv_pre_1 = nn.Conv2d(self.channel_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(self.ratio * self.compressed_size, self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_pre_2 = nn.BatchNorm2d(self.compressed_size)
        self.conv_after_1 = nn.Conv2d(self.compressed_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_after_2 = nn.Conv2d(self.ratio * self.compressed_size, self.channel_size, kernel_size=3, stride=1, padding=1)
        self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_after_2 = nn.BatchNorm2d(self.channel_size)

    def forward(self, x_raw, delta_t):
        batch, seq, channel, h, w = x_raw.shape
        if self.compressed_size != self.channel_size:
            x = F.relu(self.bn_pre_1(self.conv_pre_1(x_raw.reshape(-1,channel,h,w))))
            x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
            x = x.view(batch, seq, self.compressed_size, h, w)
        else:
            x = x_raw

        if delta_t[0] > 0:
            self.delta_t = delta_t[0]
            h = x[:,0]
            c = self.init_c
            for i in range(1, self.k):
                h,c = self.lstmcell(x[:,i], (h,c))
            for t in range(int(self.delta_t - 1)):
                print('ok')
                h,c = self.lstmcell(h, (h,c))
            if self.TM_Flag:
                w = self.time_weight(torch.cat([x[:,-1], h],1), delta_t[0])
                w = torch.tanh(0.1 * int(delta_t[0] - 1) * w)
                res = w * h + (1-w) * x[:,-1]
            else:
                res = h
        else:
            res = x[:,-1]
        if self.compressed_size != self.channel_size:
            res = F.relu(self.bn_after_1(self.conv_after_1(res)))
            res = F.relu(self.bn_after_2(self.conv_after_2(res)))
            # res = res.view(batch,channel,h,w)
        else:
            res = res
        return res.unsqueeze(1)

class MotionLSTM(nn.Module):
    def __init__(self, h,w, input_channel_size, hidden_size = 0):
        super().__init__()
        self.input_channel_size = input_channel_size  # channel size
        self.hidden_size = hidden_size
        self.h = h
        self.w = w

        #i_t 
        # self.U_i = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 

        self.U_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_i = nn.Parameter(torch.Tensor(1, self.input_channel_size, h, w))

        # #f_t 
        # self.U_f = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_f = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_f = nn.Parameter(torch.Tensor(1, self.input_channel_size, h, w))

        # #c_t 
        # self.U_c = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_c = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_c = nn.Parameter(torch.Tensor(1, self.input_channel_size, h, w))

        # #o_t 
        # self.U_o = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_o = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_o = nn.Parameter(torch.Tensor(1, self.input_channel_size, h, w))

        # self.init_weights()

    # def init_weights(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self,x,init_states=None): 
        """ 
        assumes x.shape represents (batch_size, sequence_size, input_channel_size) 
        """ 
        h, c = init_states 
        i = torch.sigmoid(self.U_i(x) + self.V_i(h) + self.b_i) 
        f = torch.sigmoid(self.U_f(x) + self.V_f(h) + self.b_f) 
        g = torch.tanh(self.U_c(x) + self.V_c(h) + self.b_c) 
        o = torch.sigmoid(self.U_o(x) + self.V_o(x) + self.b_o) 
        c_out = f * c + i * g 
        h_out = o *  torch.tanh(c_out) 

        # hidden_seq.append(h_t.unsqueeze(0)) 

        # #reshape hidden_seq p/ retornar 
        # hidden_seq = torch.cat(hidden_seq, dim=0) 
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous() 
        return (h_out, c_out)



class STPN_MotionLSTM(nn.Module):
    def __init__(self, height_feat_size = 16):
        super(STPN_MotionLSTM, self).__init__()

        # self.conv3d_1 = Conv3D(4, 8, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(8, 8, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(height_feat_size, 2*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(2*height_feat_size, 4*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(4*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(6*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(3*height_feat_size , height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn1_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn2_1 = nn.BatchNorm2d(4*height_feat_size)
        self.bn2_2 = nn.BatchNorm2d(4*height_feat_size)

        self.bn7_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn7_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn8_1 = nn.BatchNorm2d(1*height_feat_size)
        self.bn8_2 = nn.BatchNorm2d(1*height_feat_size)

    def forward(self, x):

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))
        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class Time_Modulation(nn.Module):
    def __init__(self, input_channel = 128):
        super(Time_Modulation, self).__init__()
        self.input_channel = input_channel
        self.conv1_channel = int(self.input_channel / 2)
        self.conv2_channel = int(self.conv1_channel / 2)
        # self.ratio = math.sqrt()
        self.conv1 = nn.Conv2d(self.input_channel, self.conv1_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_channel, self.conv2_channel, kernel_size=3, stride=1, padding=1)
        self.convl1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.convl2 = nn.Conv2d(8,self.conv2_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(2 * self.conv2_channel), 8, kernel_size = 3, stride=1, padding = 1) 
        # self.linear3 = nn.Linear(16,8)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1_channel)
        self.bn2 = nn.BatchNorm2d(self.conv2_channel)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(1)
        self.bnl1 = nn.BatchNorm2d(8)
        self.bnl2 = nn.BatchNorm2d(self.conv2_channel)
        self.linear3 = nn.Linear(int(2 * self.conv2_channel), 8)
        self.linear4 = nn.Linear(8, 1)
    def forward(self, x, delta_t):
        a,b,c,d = x.size()
        y = delta_t * torch.ones((a,1,c,d)).to(x.device)
        t_y = F.relu(self.bnl1(self.convl1(y)))
        t_y = F.relu(self.bnl2(self.convl2(t_y)))
        t_x = F.relu(self.bn1(self.conv1(x)))
        t_x = F.relu(self.bn2(self.conv2(t_x)))
        t_xy = torch.cat([t_x, t_y], 1)
        t_xy = F.relu(self.bn3(self.conv3(t_xy)))
        t_xy = torch.sigmoid(self.bn4(self.conv4(t_xy)))
        # t_xy = F.relu(self.bn3(self.linear3(t_xy)))
        # t_xy = torch.sigmoid(self.bn4(self.linear4(t_xy)))
        return t_xy