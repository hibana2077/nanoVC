import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import EvaBlock, EvaBlockPostNorm
from calflops import calculate_flops

# 輸入 X1 (原始音頻)(B,10,22050) X2 (特徵提取後的 represtation)(B,64,2768)
# 我想用機率密度函數與采樣去做 fusion 最後吐出 X2' (融合後音頻)(B,10,22050)

class FusionCore(nn.Module):
    def __init__(self):
        super(FusionCore, self).__init__()
        self.Adepter = nn.Sequential(
            nn.Conv1d(in_channels=212, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.GELU(),
            nn.Linear(in_features=212, out_features=3),
        )
        self.FusionCore = nn.Sequential(
            nn.Conv1d(in_channels=67, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=16),
            nn.GELU(),
            nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x_comb):
        x1, x2 = x_comb
        # X1 -> (B,3,22050)
        # X2 -> (B,212,212)
        x2_adp = self.Adepter(x2)
        # X2' -> (B,64,3)
        x_dot = x2_adp @ x1
        # X12 -> (B,64,22050)
        x = torch.cat((x1, x_dot), dim=1)
        # X12 -> (B,67,22050)
        out = self.FusionCore(x)
        # X12' -> (B,3,22050)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FusionCore().to(device)
    X1 = torch.rand(1, 3, 22050).to(device)
    X2 = torch.rand(1, 212, 212).to(device)
    ts = time.time()
    output = model((X1, X2))
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")