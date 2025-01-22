import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.ghostnet import _SE_LAYER
from calflops import calculate_flops

# 輸入 X1 (原始音頻)(B,10,22050) X2 (特徵提取後的 represtation)(B,64,2768)
# 我想用機率密度函數與采樣去做 fusion 最後吐出 X2' (融合後音頻)(B,10,22050)

class GhostModule(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            use_act=True,
            act_layer=nn.ReLU,
    ):
        super(GhostModule, self).__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm1d(init_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_chs, new_chs, dw_size, 1, dw_size//2, groups=init_chs, bias=False),
            nn.BatchNorm1d(new_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            act_layer=nn.ReLU,
            se_ratio=0.,
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.ghost1 = GhostModule(in_chs, mid_chs, use_act=True, act_layer=act_layer)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv1d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm1d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, use_act=False)
        
        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm1d(in_chs),
                nn.Conv1d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_chs),
            )

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(shortcut)
        return x

class FusionCore(nn.Module):
    def __init__(self):
        super(FusionCore, self).__init__()

        self.ghost_fusion = nn.Sequential(
            GhostBottleneck(6, 128, 5,act_layer=nn.LeakyReLU),
            GhostBottleneck(5, 64, 4,act_layer=nn.LeakyReLU),
            GhostBottleneck(4, 32, 3,act_layer=nn.LeakyReLU),
            GhostBottleneck(3, 16, 3,act_layer=nn.LeakyReLU),
            nn.Tanh()
        )

    def forward(self, x_comb):
        """
        x_comb: (x1, x2f)
        x1: (B, 3, T)
        x2f: (B, 3, T)
        """

        x1, x2f = x_comb

        # x12: (B, 3, T) -> (B, 6, T)
        x12 = torch.cat([x1, x2f], dim=1)

        # x12: (B, 6, T) -> (B, 3, T)
        x12 = self.ghost_fusion(x12)

        return x12


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FusionCore().to(device)
    X1 = torch.rand(1, 3, 16000).to(device)
    X2 = torch.rand(1, 3, 16000).to(device)
    ts = time.time()
    output = model((X1, X2))
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")