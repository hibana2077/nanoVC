import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import EvaBlockPostNorm
from calflops import calculate_flops    

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # 下採樣區塊 1
        self.down1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        # 下採樣區塊 2
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        # 下採樣區塊 3
        self.down3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )

        # CONV Transpose for EvaBlocks
        self.adjust = nn.Conv1d(2757, 256, kernel_size=1, stride=1, padding=0)

        self.EvaBlocks = nn.Sequential(
            EvaBlockPostNorm(dim=256, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
            EvaBlockPostNorm(dim=256, num_heads=16, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
            EvaBlockPostNorm(dim=256, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
        )

    def forward(self, x):
        # x shape: (B, 3, T)
        d1 = self.down1(x)   # (B, 32, T/2)
        d2 = self.down2(d1)  # (B, 64, T/4)
        d3 = self.down3(d2)  # (B, 128, T/8)
        
        # Transpose for EvaBlocks
        d3 = d3.transpose(1, 2)  # (B, 256, 256)
        d3 = self.adjust(d3)
        d3 = d3.transpose(1, 2)  # (B, 256, 256)

        b = self.EvaBlocks(d3)  # (B, 256, T/8)
        
        # 回傳每層的特徵，供 Decoder 做 skip connection
        return [d1, d2, d3, b]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor()
    # model = EvaBlockPostNorm(dim=64, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
    batch_size = 1
    input_size = (batch_size, 3, 22050)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_size,
                                        output_as_string=True,
                                        print_results=False,
                                        output_precision=4)
    print("FeatureExtractor FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    model.to(device)
    x = torch.rand(1, 3, 22050).to(device)
    # x = torch.rand(1, 64, 128).to(device)
    ts = time.time()
    output = model(x)
    te = time.time()
    print(f"Time taken: {te-ts:.2f} seconds")
    print(f"Output shape: {output[0].shape}")
    print(f"Output shape: {output[1].shape}")
    print(f"Output shape: {output[2].shape}")
    print(f"Output shape: {output[3].shape}")
    # FeatureExtractor FLOPs:2.7155 GFLOPS   MACs:1.3546 GMACs   Params:3.2598 M

    # Time taken: 0.17 seconds
    # Output shape: torch.Size([1, 32, 11025])
    # Output shape: torch.Size([1, 128, 5513])
    # Output shape: torch.Size([1, 256, 256])
    # Output shape: torch.Size([1, 256, 256])