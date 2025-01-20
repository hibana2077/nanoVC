import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from calflops import calculate_flops

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.Compressor = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=32, kernel_size=4, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.ReLU()
        )
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2758, nhead=14, batch_first=True), num_layers=3)

    def forward(self, x):
        x = self.Compressor(x)
        x = self.TransformerEncoder(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor()
    batch_size = 1
    input_size = (batch_size, 10, 22050)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_size,
                                        output_as_string=True,
                                        print_results=False,
                                        output_precision=4)
    print("FeatureExtractor FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    model.to(device)
    x = torch.rand(1, 10, 22050).to(device)
    ts = time.time()
    output = model(x)
    te = time.time()
    print(f"Time taken: {te-ts:.2f} seconds, Output shape: {output.shape}")
    # FeatureExtractor FLOPs:5.391 GFLOPS   MACs:2.6943 GMACs   Params:41.7574 M 

    # Time taken: 0.17 seconds, Output shape: torch.Size([1, 64, 2758])