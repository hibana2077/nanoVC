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
            nn.Conv1d(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Linear(in_features=2758, out_features=22050),
        )
        # self.FusionCore = nn.TransformerDecoderLayer(d_model=22050, nhead=5, batch_first=True)
        self.FusionCore = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.Linear(in_features=22050, out_features=22050),
            nn.SiLU(),
            nn.Linear(in_features=22050, out_features=22050),
        )

    def forward(self, x_comb):
        x1, x2 = x_comb
        x2_adp = self.Adepter(x2)
        x = torch.cat((x1, x2_adp), dim=1)
        # out = self.FusionCore(x1, x)
        out = self.FusionCore(x)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = FusionCore().to(device)
    X1 = torch.rand(1, 10, 22050).to(device)
    X2 = torch.rand(1, 64, 2758).to(device)
    ts = time.time()
    output = model((X1, X2))
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")