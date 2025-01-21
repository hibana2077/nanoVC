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
        
        # 假設要把 (B,212,T/8) 還原回 (B,3,T)
        # 可使用 Upsample+Conv 或 ConvTranspose1d
        
        # Up Block 1：從 212 -> 128
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.d3_adj_a = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        # Up Block 2：從 128 -> 64
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.d2_adj_a = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.d2_adj_b = nn.Sequential(
            nn.Conv1d(5513, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        # Up Block 3：從 64 -> 32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.d1_adj_a = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.d1_adj_b = nn.Sequential(
            nn.Conv1d(11025, 2048, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        # 最終還原到 3 channel (對應原始波形 channel 數)
        self.final_conv = nn.Conv1d(32, 3, kernel_size=3, padding=1)
        self.recover_conv = nn.Conv1d(2048, 22050, kernel_size=3, padding=1)

        self.ghost_fusion = GhostBottleneck(6, 128, 3,act_layer=nn.LeakyReLU)

    def forward(self, x_comb):
        """
        x_comb: (x1, x2f)
        x1: (B, 3, T) - 這裡視需求看要不要參與Decoder運算
        x2f: [d1, d2, d3, bottleneck] 來自Encoder (FeatureExtractor)
        """
        x1, x2f = x_comb
        d1, d2, d3, bottleneck = x2f  # 取出每層特徵
        
        # 逐層上採樣
        x = self.up1(bottleneck)  # (B,128, 512)
        d3 = self.d3_adj_a(d3)      # (B,128, 512)
        x = x + d3
        
        x = self.up2(x)           # (B,64, T/2)
        d2 = self.d2_adj_a(d2)    # (B,64, T/2)
        d2 = d2.transpose(1, 2)   # (B, T/2, 64)
        d2 = self.d2_adj_b(d2)    # (B, T/2, 1024)
        d2 = d2.transpose(1, 2)
        x = x + d2
        
        x = self.up3(x)           # (B,32, T)
        d1 = self.d1_adj_a(d1)    # (B,32, T)
        d1 = d1.transpose(1, 2)   # (B, T, 32)
        d1 = self.d1_adj_b(d1)    # (B, T, 1024)
        d1 = d1.transpose(1, 2)
        x = x + d1
        
        # 最終輸出 (B,3,T)
        x = self.final_conv(x)
        
        x = x.transpose(1, 2)
        x = self.recover_conv(x)
        x = x.transpose(1, 2)

        # Fusion (1D-GhostNet)

        x_comb = torch.cat([x1, x], dim=1)
        out = self.ghost_fusion(x_comb)
        out = F.tanh(out)
        
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FusionCore().to(device)
    X1 = torch.rand(1, 3, 22050).to(device)
    X2 = [
        torch.rand(1, 32, 11025).to(device),
        torch.rand(1, 128, 5513).to(device),
        torch.rand(1, 256, 256).to(device),
        torch.rand(1, 256, 256).to(device)
    ]
    ts = time.time()
    output = model((X1, X2))
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")