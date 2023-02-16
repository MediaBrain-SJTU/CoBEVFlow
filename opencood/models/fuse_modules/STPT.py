
import numpy as np
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import ipdb
import copy
import matplotlib.pyplot as plt
import os

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def window_partition(x, window_size):
    """Partition the input video sequences into several windows along spatial 
    dimensions.
    Args:
        x (torch.Tensor): (B, D, H, W, C)
        window_size (tuple[int]): Window size
    Returns: 
        windows: (B*nW, D, Wh, Ww, C)
    """
    B, D, H, W, C = x.shape
    # B, D, num_Hwin, Wh, num_Wwin, Ww, C
    x = x.view(B, D, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C) 
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, D, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """Reverse window partition.
    Args:
        windows (torch.Tensor): (B*nW, D, Wh, Ww, C)
        window_size (tuple[int]): Window size
        B (int): Number of batches
        D (int): Number of frames
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1], D, window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    """Adjust window size and shift size based on the size of the input.
    Args:
        x_size (tuple[int]): The shape of x.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int], optional): Shift size. Defaults to None.
    Returns:
        use_window_size: Window size for use.
        use_shift_size: Shift size for use.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention3D(nn.Module):
    """Window based multi-head self/cross attention (W-MSA/W-MCA) module with relative 
    position bias. 
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim, num_frames_q, num_frames_kv, window_size, num_heads, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialization function.
        Args:
            dim (int): Number of input channels.
            num_frames (int): Number of input frames.
            window_size (tuple[int]): The size of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Dropout ratio of attention weight. Defaults to 0.0
            proj_drop (float, optional): Dropout ratio of output. Defaults to 0.0
        """
        super().__init__()
        self.dim = dim
        self.num_frames_q = num_frames_q # D1
        self.num_frames_kv = num_frames_kv # D2
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads # nH
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_frames_q - 1) * (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*D-1 * 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_d_q = torch.arange(self.num_frames_q)
        # print(self.num_frames_q,self.num_frames_kv)
        if int((self.num_frames_q + 1) // self.num_frames_kv) != 0:
            coords_d_kv = torch.arange(0, self.num_frames_q, int((self.num_frames_q + 1) // self.num_frames_kv))
        else:
            coords_d_kv = torch.tensor([0]*self.num_frames_kv)
        # print(coords_d_kv,'coords_d_kv')
        # coords_d_kv = torch.tensor([0])
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_q = torch.stack(torch.meshgrid([coords_d_q, coords_h, coords_w]))  # 3, D2, Wh, Ww
        coords_kv = torch.stack(torch.meshgrid([coords_d_kv, coords_h, coords_w]))  # 3, D1, Wh, Ww
        # print('coords_q',coords_q.shape,'coords_kv',coords_kv.shape)
        coords_q_flatten = torch.flatten(coords_q, 1)  # 3, D1*Wh*Ww
        coords_kv_flatten = torch.flatten(coords_kv, 1)  # 3, D2*Wh*Ww
        relative_coords = coords_q_flatten[:, :, None] - coords_kv_flatten[:, None, :]  # 3, D1*Wh*Ww, D2*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # D1*Wh*Ww, D2*Wh*Ww, 3
        relative_coords[:, :, 0] += self.num_frames_q - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 2] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # D1*Wh*Ww, D2*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        """Forward function.
        Args:
            q (torch.Tensor): (B*nW, D1*Wh*Ww, C)
            kv (torch.Tensor): (B*nW, D2*Wh*Ww, C). Defaults to None.
            mask (torch.Tensor, optional): Mask for shifted window attention (nW, D1*Wh*Ww, D2*Wh*Ww). Defaults to None.
        Returns:
            torch.Tensor: (B*nW, D1*Wh*Ww, C)
        """
        # ipdb.set_trace()
        kv = q if kv is None else kv
        B_, N1, C = q.shape # N1 = D1*Wh*Ww, B_ = B*nW
        B_, N2, C = kv.shape # N2 = D2*Wh*Ww, B_ = B*nW
        # print('N1',N1,'N2',N2)
        q = self.q(q).reshape(B_, N1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B_, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1] # B_, nH, N1(2), C
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # B_, nH, N1, N2

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N1, N2, -1)  # D1*Wh*Ww, D2*Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, D1*Wh*Ww, D2*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, D1*Wh*Ww, D2*Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0) # B, nW, nH, D1*Wh*Ww, D2*Wh*Ww
            attn = attn.view(-1, self.num_heads, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class VSTSREncoderTransformerBlock(nn.Module):
    """Video spatial-temporal super-resolution encoder transformer block.
    """
    def __init__(self, dim, num_heads, num_frames=4, window_size=(8, 8), 
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,encoder_id = 0, layer_id = 0):
        """Initialization function.
        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            num_frames (int): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, num_frames_q=self.num_frames, num_frames_kv=self.num_frames,
            window_size=self.window_size, num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.encoder_id = encoder_id
        self.layer_id = layer_id

    def forward(self, x, mask_matrix, data_id = 0):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, H, W, C)
            mask_matrix (torch.Tensor): (nW*B, D*Wh*Ww, D*Wh*Ww)
        Returns:
            torch.Tensor: (B, D, H, W, C)
        """
        # ipdb.set_trace()
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D, window_size, window_size, C
        x_windows = x_windows.view(-1, D * window_size[0] * window_size[1], C)  # nW*B, D*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)[0]  # nW*B, D*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, D, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, D, Hp, Wp)  # B, D, H, W, C
        # save_dir = '/dssg/home/acct-eezy/eezy-user1/wz/multi-cam-pnp_11/vis_stpt/{}/'.format(data_id)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        
        # attention_matrix = copy.deepcopy(shifted_x).cpu()
        # for t in range(3):
        #     feature_map = attention_matrix[0, t]
        #     max_feature_map = torch.max(feature_map, -1)[0]
            
        #     plt.matshow(-max_feature_map.numpy(), cmap=plt.cm.Blues)
        #     plt.grid(False)
        #     # plt.imshow(-max_feature_map.numpy())
        #     plt.savefig(save_dir + 'encoder_{}_depth_{}_t_{}.png'.format(self.encoder_id, self.layer_id, t), bbox_inches='tight')
        # torch.save(shifted_x, save_dir + 'encoder_{}_depth_{}.pt'.format(self.encoder_id, self.layer_id))
        # ipdb.set_trace()

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class VSTSRDecoderTransformerBlock(nn.Module):
    """Video spatial-temporal super-resolution decoder transformer block.
    """
    def __init__(self, dim, num_heads, num_frames=4, num_out_frames=5, window_size=(8, 8), 
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        """Initialization function.
        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            num_frames (int): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        # self.num_out_frames = 2 * num_frames - 1
        self.num_out_frames = num_out_frames
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention3D(
            dim, num_frames_q=self.num_out_frames, 
            num_frames_kv=self.num_out_frames, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.attn2 = WindowAttention3D(
            dim, num_frames_q=self.num_out_frames, 
            num_frames_kv=self.num_frames, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
       

    def forward(self, x, attn_kv, mask_matrix_q, mask_matrix_qkv):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D1, H, W, C)
            attn_kv (torch.Tensor): (B, D2, H, W, C)
            mask_matrix_q (torch.Tensor): (nW*B, D1*Wh*Ww, D1*Wh*Ww)
            mask_matrix_qkv (torch.Tensor): (nW*B, D1*Wh*Ww, D2*Wh*Ww)
        Returns:
            torch.Tensor: (B, D1, H, W, C)
        """
        B, D1, H, W, C = x.shape
        B, D2, H, W, C = attn_kv.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        # shortcut = x
        # x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        # x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        # _, _, Hp, Wp, _ = x.shape

        # # cyclic shift
        # if any(i > 0 for i in shift_size):
        #     shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
        #     attn_mask_q = mask_matrix_q
        #     attn_mask_qkv = mask_matrix_qkv
        # else:
        #     shifted_x = x
        #     attn_mask_q = None
        #     attn_mask_qkv = None

        # # partition windows
        # x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
        # x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C

        # # W-MSA/SW-MSA for query
        # attn_windows = self.attn1(x_windows, mask=attn_mask_q)[0] # nW*B, D1*window_size*window_size, C
        # attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
        # shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp) # B, D1, Hp, Wp, C

        # # reverse cyclic shift
        # if any(i > 0 for i in shift_size):
        #     x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        # else:
        #     x = shifted_x
        
        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :, :H, :W, :].contiguous()

        # x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        attn_kv = self.norm_kv(attn_kv)
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape
        attn_kv = F.pad(attn_kv, (0, 0, 0, pad_r, 0, pad_b, 0, 0))

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            shifted_attn_kv = torch.roll(attn_kv, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask_q = mask_matrix_q
            attn_mask_qkv = mask_matrix_qkv
        else:
            shifted_x = x
            shifted_attn_kv = attn_kv
            attn_mask_q = None
            attn_mask_qkv = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
        attn_kv_windows = window_partition(shifted_attn_kv, window_size)  # nW*B, D2, window_size, window_size, C
        x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C
        attn_kv_windows = attn_kv_windows.view(-1, D2 * window_size[0] * window_size[1], C)  # nW*B, D2*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn2(x_windows, attn_kv_windows, mask=attn_mask_qkv)[0]  # nW*B, D1*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp)  # B, D1, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x

# class VSTSRDecoderTransformerBlock(nn.Module):
#     """Video spatial-temporal super-resolution decoder transformer block.
#     """
#     def __init__(self, dim, num_heads, num_frames=4, num_out_frames=5, window_size=(8, 8), 
#                  shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
#                  norm_layer=nn.LayerNorm):
#         """Initialization function.

#         Args:
#             dim (int): Number of input channels. 
#             num_heads (int): Number of attention heads.
#             num_frames (int): Number of input frames.
#             window_size (tuple[int], optional): Window size. Defaults to 8.
#             shift_size (tuple[int], optional): Shift size. Defaults to 0.
#             mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
#             qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
#             qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
#             drop (float, optional): Dropout rate. Defaults to 0.
#             attn_drop (float, optional): Attention dropout rate. Defaults to 0.
#             drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
#             act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
#             norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
#         """
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.num_frames = num_frames
#         # self.num_out_frames = 2 * num_frames - 1
#         self.num_out_frames = num_out_frames
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
#         assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

#         self.norm1 = norm_layer(dim)
#         self.attn1 = WindowAttention3D(
#             dim, num_frames_q=self.num_out_frames, 
#             num_frames_kv=self.num_out_frames, window_size=self.window_size, 
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
#             attn_drop=attn_drop, proj_drop=drop
#         )
#         self.attn2 = WindowAttention3D(
#             dim, num_frames_q=self.num_out_frames, 
#             num_frames_kv=self.num_frames, window_size=self.window_size, 
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
#             attn_drop=attn_drop, proj_drop=drop
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.norm3 = norm_layer(dim)
#         self.norm_kv = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       

#     def forward(self, x, attn_kv, mask_matrix_q, mask_matrix_qkv):
#         """Forward function.

#         Args:
#             x (torch.Tensor): (B, D1, H, W, C)
#             attn_kv (torch.Tensor): (B, D2, H, W, C)
#             mask_matrix_q (torch.Tensor): (nW*B, D1*Wh*Ww, D1*Wh*Ww)
#             mask_matrix_qkv (torch.Tensor): (nW*B, D1*Wh*Ww, D2*Wh*Ww)

#         Returns:
#             torch.Tensor: (B, D1, H, W, C)
#         """
#         B, D1, H, W, C = x.shape
#         B, D2, H, W, C = attn_kv.shape
#         window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

#         shortcut = x
#         x = self.norm1(x)

#         # Padding
#         pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
#         pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
#         x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
#         _, _, Hp, Wp, _ = x.shape

#         # cyclic shift
#         if any(i > 0 for i in shift_size):
#             shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
#             attn_mask_q = mask_matrix_q
#             attn_mask_qkv = mask_matrix_qkv
#         else:
#             shifted_x = x
#             attn_mask_q = None
#             attn_mask_qkv = None

#         # partition windows
#         x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
#         x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C

#         # W-MSA/SW-MSA for query
#         attn_windows = self.attn1(x_windows, mask=attn_mask_q)[0] # nW*B, D1*window_size*window_size, C
#         attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
#         shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp) # B, D1, Hp, Wp, C

#         # reverse cyclic shift
#         if any(i > 0 for i in shift_size):
#             x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
#         else:
#             x = shifted_x
        
#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :, :H, :W, :].contiguous()

#         x = shortcut + self.drop_path(x)

#         shortcut = x
#         x = self.norm2(x)
#         attn_kv = self.norm_kv(attn_kv)
#         x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
#         attn_kv = F.pad(attn_kv, (0, 0, 0, pad_r, 0, pad_b, 0, 0))

#         # cyclic shift
#         if any(i > 0 for i in shift_size):
#             shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
#             shifted_attn_kv = torch.roll(attn_kv, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
#             attn_mask_q = mask_matrix_q
#             attn_mask_qkv = mask_matrix_qkv
#         else:
#             shifted_x = x
#             shifted_attn_kv = attn_kv
#             attn_mask_q = None
#             attn_mask_qkv = None

#         # partition windows
#         x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
#         attn_kv_windows = window_partition(shifted_attn_kv, window_size)  # nW*B, D2, window_size, window_size, C
#         x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C
#         attn_kv_windows = attn_kv_windows.view(-1, D2 * window_size[0] * window_size[1], C)  # nW*B, D2*window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn2(x_windows, attn_kv_windows, mask=attn_mask_qkv)[0]  # nW*B, D1*window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
#         shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp)  # B, D1, H, W, C

#         # reverse cyclic shift
#         if any(i > 0 for i in shift_size):
#             x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :, :H, :W, :].contiguous()

#         x = shortcut + self.drop_path(x)

#         # FFN
#         x = x + self.drop_path(self.mlp(self.norm3(x)))

#         return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, num_frames, window_size=(8, 8), 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, encoder_id = 0):
        """Encoder layer
        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            num_frames (int]): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.encoder_id = encoder_id
        # Build blocks
        self.blocks = nn.ModuleList([
            VSTSREncoderTransformerBlock(dim=dim, num_heads=num_heads,
            num_frames=num_frames,window_size=window_size, 
            shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            encoder_id = self.encoder_id,
            layer_id = i)
        for i in range(depth)])


    def forward(self, x, data_id = 0):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, C, H, W)
        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        # ipdb.set_trace()
        B, D, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2) # B, D, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask = torch.zeros((1, D, Hp, Wp, 1), device=x.device) # 1, D, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size) # nW, D, Wh, Ww, 1
        mask_windows = mask_windows.view(-1, D * window_size[0] * window_size[1]) # nW, D*Wh*Ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW, D*Wh*Ww, D*Wh*Ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask, data_id)

        x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W

        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, num_frames, num_out_frames, window_size=(8, 8), 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Decoder layer
        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            num_frames (int]): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            VSTSRDecoderTransformerBlock(dim=dim, num_heads=num_heads,
            num_frames=num_frames,num_out_frames=num_out_frames, window_size=window_size, 
            shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)
        for i in range(depth)])

    def forward(self, x, attn_kv):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D1, C, H, W)
            attn_kv (torch.Tensor): (B, D2, C, H, W)
        Returns:
            torch.Tensor: (B, D1, C, H, W)
        """
        B, D1, C, H, W = x. shape
        _, D2, C, _, _ = attn_kv.shape
        x = x.permute(0, 1, 3, 4, 2) # B, D1, H, W, C
        attn_kv = attn_kv.permute(0, 1, 3, 4, 2) # B, D2, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask_q = torch.zeros((1, D1, Hp, Wp, 1), device=x.device) # 1, D1, H, W, 1
        img_mask_kv = torch.zeros((1, D2, Hp, Wp, 1), device=x.device) # 1, D2, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask_q[:, :, h, w, :] = cnt
                img_mask_kv[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows_q = window_partition(img_mask_q, window_size) # nW, D1, Wh, Ww, 1
        mask_windows_kv = window_partition(img_mask_kv, window_size) # nW, D2, Wh, Ww, 1
        mask_windows_q = mask_windows_q.view(-1, D1 * window_size[0] * window_size[1]) # nW, D1*Wh*Ww
        mask_windows_kv = mask_windows_kv.view(-1, D2 * window_size[0] * window_size[1]) # nW, D2*Wh*Ww
        attn_mask_q = mask_windows_q.unsqueeze(1) - mask_windows_q.unsqueeze(2) # nW, D1*Wh*Ww, D1*Wh*Ww
        attn_mask_qkv = mask_windows_kv.unsqueeze(1) - mask_windows_q.unsqueeze(2) # nW, D1*Wh*Ww, D2*Wh*Ww
        attn_mask_q = attn_mask_q.masked_fill(attn_mask_q != 0, float(-100.0)).masked_fill(attn_mask_q == 0, float(0.0))
        attn_mask_qkv = attn_mask_qkv.masked_fill(attn_mask_qkv != 0, float(-100.0)).masked_fill(attn_mask_qkv == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_kv, attn_mask_q, attn_mask_qkv)

        x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W

        return x


class InputProj(nn.Module):
    """Video input projection
    Args:
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        kernel_size (int): Size of the convolution kernel. Default: 3
        stride (int): Stride of the convolution. Default: 1
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.LeakyReLU.
    """
    def __init__(self, in_channels=3, embed_dim=32, kernel_size=3, stride=1, 
                 norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()

        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, D, C, H, W)
        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.reshape(-1, C, H, W) # B*D, C, H, W
        x = self.proj(x).reshape(B, D, -1, H, W) # B, D, C, H, W
        if self.norm is not None:
            x = x.permute(0, 1, 3, 4, 2) # B, D, H, W, C
            x = self.norm(x)
            x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W
        return x

class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, C, H, W)
        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.view(-1, C, H, W) # B*D, C, H, W
        out = self.conv(x).view(B, D, -1, H // 2, W // 2)  # B, D, C, H, W
        return out

class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, C, H, W)
        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x. shape
        x = x.view(-1, C, H, W) # B*D, C, H, W
        out = self.deconv(x).view(B, D, -1, H * 2, W * 2) # B, D, C, H, W
        return out


class HDmapConv(nn.Module):
    def __init__(self, in_channels=2, out_channels=144):
        super().__init__()
        self.hdmap_encoder=nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, hdmap_x):
        """
        hdmap_x: b,2,200,200
        output: b,out_c,25,25
        """
        return self.hdmap_encoder(hdmap_x)


class HDmapfeatConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=144):
        super().__init__()
        self.hdmap_feat_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, hdmap_x):
        """
        hdmap_x: b,256,200,200
        output: b,out_c,25,25
        """
        return self.hdmap_feat_encoder(hdmap_x)


class STPT(nn.Module):
    # def __init__(self, in_chans=3, embed_dim=96, 
    #              depths=[8, 8, 8, 8, 8, 8, 8, 8], 
    #              num_heads=[2, 4, 8, 16, 16, 8, 4, 2], num_frames=4,
    #              window_sizes=[(4,4), (4,4), (4,4), (4,4), (4,4), (4,4), (4,4), (4,4)], 
    #              mlp_ratio=2., qkv_bias=True, qk_scale=None,
    #              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #              norm_layer=nn.LayerNorm, patch_norm=True, 
    #              back_RBs=0):
    def __init__(self, in_chans=3, embed_dim=96, 
                 depths=[4, 4, 4, 4, 4, 4, 4, 4], 
                 num_heads=[2, 4, 8, 16, 16, 8, 4, 2], num_frames=4, num_out_frames=5,
                 window_sizes=[(2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2)], 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 back_RBs=0, use_hdmap=False, use_hdmap_feat=False):

    
        """
        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Number of projection output channels. Defaults to 32.
            depths (list[int], optional): Depths of each Transformer stage. Defaults to [2, 2, 2, 2, 2, 2, 2, 2].
            num_heads (list[int], optional): Number of attention head of each stage. Defaults to [2, 4, 8, 16, 16, 8, 4, 2].
            num_frames (int, optional): Number of input frames. Defaults to 4.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            back_RBs (int, optional): Number of residual blocks for super resolution. Defaults to 10.
            use_hdmap: use hdmap for bev query
        """
        super().__init__()
        print(depths)
        print(num_heads)
        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers - 1)
        dec_depths = depths[self.num_enc_layers:]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_in_frames = num_frames
        # self.num_out_frames = 2 * num_frames - 1
        self.num_out_frames = num_out_frames

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth 
        enc_dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(in_channels=in_chans, embed_dim=embed_dim,
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            encoder_layer = EncoderLayer(
                    dim=embed_dim, 
                    depth=depths[i_layer], num_heads=num_heads[i_layer], 
                    num_frames=num_frames, window_size=window_sizes[i_layer], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=enc_dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    encoder_id = i_layer
            )
            downsample = Downsample(embed_dim, embed_dim)
            self.encoder_layers.append(encoder_layer)
            self.downsample.append(downsample)


        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = DecoderLayer(
                    dim=embed_dim, 
                    depth=depths[i_layer + self.num_enc_layers], 
                    num_heads=num_heads[i_layer + self.num_enc_layers], 
                    num_frames=num_frames, num_out_frames = num_out_frames, window_size=window_sizes[i_layer + self.num_enc_layers], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dec_dpr[sum(dec_depths[:i_layer]):sum(dec_depths[:i_layer + 1])],
                    norm_layer=norm_layer
            )
            self.decoder_layers.append(decoder_layer)
            if i_layer != self.num_dec_layers - 1:
                upsample = Upsample(embed_dim, embed_dim)
                self.upsample.append(upsample)

        # Reconstruction block
        # ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=embed_dim)
        # self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        # Upsampling
        # self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # hdmap query encoder
        self.use_hdmap = use_hdmap
        self.use_hdmap_feat = use_hdmap_feat
        if self.use_hdmap and not self.use_hdmap_feat:
            self.hdmap_conv = HDmapConv(out_channels=self.embed_dim)
        elif self.use_hdmap_feat:
            self.hdmap_feat_conv = HDmapfeatConv(out_channels=self.embed_dim)
        
        self.future_query = nn.Parameter(torch.Tensor(1, self.num_out_frames, self.embed_dim, 100 // self.scale, 352 // self.scale), requires_grad=True)
        # self.future_query = nn.Parameter(torch.Tensor(1, 1, self.embed_dim, 25, 25), requires_grad=True)
        nn.init.xavier_uniform_(self.future_query)
        
        self.col_embed = nn.Embedding(25, self.embed_dim // 3)
        self.row_embed = nn.Embedding(25, self.embed_dim // 3)
        self.t_embed = nn.Embedding(self.num_out_frames, self.embed_dim // 3)
        
        self.my_conv = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, hdmap=None, data_id = 0):
        """
        hdmap: if hdmap not none, use hdmap feat to bev query
        """
        B, D, C, H, W = x.size()  # D input video frames
        # x = x.permute(0, 2, 1, 3, 4)
        # upsample_x = F.interpolate(x, (2*D-1, H*4, W*4), mode='trilinear', align_corners=False)
        # x = x.permute(0, 2, 1, 3, 4)
        # ipdb.set_trace()
        x = self.input_proj(x) # B, D, C, H, W

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))

        encoder_features = []
        for i_layer in range(self.num_enc_layers):
            x = self.encoder_layers[i_layer](x, data_id)
            encoder_features.append(x)
            if i_layer != self.num_enc_layers - 1:
                x = self.downsample[i_layer](x)
        # ipdb.set_trace()
        _, _, C, h, w = x.size()
        # print(x.shape)
        #####################################################################################
        # TODO: Use interpolation for queries
        # y = torch.zeros((B, self.num_out_frames, C, h, w), device=x.device)
        # for i in range(self.num_out_frames):
        #     if i % 2 == 0:
        #         y[:, i, :, :, :] = x[:, i//2]
        #     else:
        #         y[:, i, :, :, :] = (x[:, i//2] + x[:, i//2 + 1]) / 2
        #####################################################################################
        
        
        x_grid = torch.arange(25, device = x.device)
        y_grid = torch.arange(25, device = x.device)
        t_grid = torch.arange(self.num_out_frames, device = x.device)
        
        x_embed = self.col_embed(x_grid)
        y_embed = self.row_embed(y_grid)
        t_embed = self.t_embed(t_grid)
        
        query_embed = torch.cat((
                                x_embed.unsqueeze(0).unsqueeze(0).repeat(self.num_out_frames, 25, 1, 1),
                                y_embed.unsqueeze(1).unsqueeze(0).repeat(self.num_out_frames, 1, 25, 1),
                                t_embed.unsqueeze(1).unsqueeze(1).repeat(1, 25, 25, 1)),
                                dim = -1
                                ).permute(0, 3, 1, 2).repeat(B, 1, 1, 1, 1)
        
        # print("query_embed", query_embed.shape)  # 1,5,144,25,25
        # print("future_query", self.future_query.shape)  # 1,5,144,25,25
        
        # y = self.future_query.repeat(B, self.num_out_frames, 1, 1, 1)
        y = self.future_query.repeat(B, 1, 1, 1, 1)
        # print('y', y.shape)  # 1,5,144,25,25

        # y = y + query_embed

        if hdmap is not None:
            if not self.use_hdmap_feat:
                hdmap_feat = self.hdmap_conv(hdmap)  # b, 144, 25, 25
                hdmap_feat = hdmap_feat.unsqueeze(dim=1).repeat(1, self.num_out_frames, 1, 1, 1)
                y = y + hdmap_feat
            else:
                hdmap_feat = self.hdmap_feat_conv(hdmap)
                hdmap_feat = hdmap_feat.unsqueeze(dim=1).repeat(1, self.num_out_frames, 1, 1, 1)
                y = y + hdmap_feat
        
        for i_layer in range(self.num_dec_layers):
            y = self.decoder_layers[i_layer](y, encoder_features[-i_layer - 1])
            if i_layer != self.num_dec_layers - 1:
                y = self.upsample[i_layer](y)
        # ipdb.set_trace()
        y = y[:, :, :, :H, :W].contiguous()
        # Super-resolution
        B, D, C, H, W = y.size()
        y = y.view(B*D, C, H, W)
        ############################################################################
        # out = self.recon_trunk(y)
        # out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        # out = self.lrelu(self.HRconv(out))
        # out = self.conv_last(out)
        ############################################################################
        
        out = self.my_conv(y)
        _, _, H, W = out.size()
        outs = out.view(B, self.num_out_frames, -1, H, W)
        outs = outs# + upsample_x.permute(0, 2, 1, 3, 4)
        return outs
    
    
    
if __name__ == '__main__':
    device = torch.device('cuda',7)
    a = torch.randn((4, 3, 128, 100, 352)).to(device)
    model = STPT(in_chans=128,
                depths=[4, 4], 
                num_heads=[2, 2], 
                window_sizes=[(4,4), (4,4)],
                num_frames=3,
                num_out_frames=1,
                embed_dim=48,
                use_hdmap=False,
                use_hdmap_feat=False).to(device)
    
    output = model(a)
    print(output.shape)     #[1, 5, 32, 200, 200])