import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import EvaBlockPostNorm
from calflops import calculate_flops

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.Compressor = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, stride=7, padding=7),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=212, kernel_size=3, stride=3, padding=3),
            nn.LeakyReLU()
        )
        self.EvaBlocks = nn.Sequential(
            EvaBlockPostNorm(dim=212, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
            EvaBlockPostNorm(dim=212, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
            EvaBlockPostNorm(dim=212, num_heads=4, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm),
        )

    def forward(self, x):
        x = self.Compressor(x)
        x = self.EvaBlocks(x)
        return x
    

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
    print(f"Time taken: {te-ts:.2f} seconds, Output shape: {output.shape}")
    # FeatureExtractor FLOPs:274 MFLOPS   MACs:136.499 MMACs   Params:595.364 K

    # Time taken: 0.17 seconds, Output shape: torch.Size([1, 64, 2758])