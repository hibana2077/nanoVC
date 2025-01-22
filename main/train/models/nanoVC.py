import pprint
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from transformers import AutoFeatureExtractor
from .fusion_core import FusionCore

class NanoVC(nn.Module):
    def __init__(self, Training: bool = False):
        super(NanoVC, self).__init__()
        self.FeatureExtractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Freeze FeatureExtractor
        for param in self.FeatureExtractor.parameters():
            param.requires_grad = False
        
        self.FusionCore = FusionCore()
        self.Training = Training

    def forward(self, x1, x2):
        x2f = self.FeatureExtractor(x2)
        x_sy = self.FusionCore((x1, x2f))
        if self.Training:
            x_syf = self.FeatureExtractor(x_sy)
            return x_sy, x_syf, x2f
        return x_sy

    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = NanoVC().to(device)
    X1 = torch.rand(1, 10, 22050).to(device)
    X2 = torch.rand(1, 10, 22050).to(device)
    ts = time.time()
    output = model(X1, X2)
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")

    # Fvcore Flops
    fva = FlopCountAnalysis(model, (X1, X2))
    print(f"Flops: {fva.total()/1e9:.2f} GFLOPs")

    # params = parameter_count(model)
    print(parameter_count_table(model))