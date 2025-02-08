import torch
from torch import nn
from transformers.activations import ACT2FN

class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=2048, hidden_act="gelu_fast"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

if __name__ == "__main__":
    model = DeepseekV3MLP()
    x = torch.randn(1, 2048)
    y = model(x)
    print(y.shape)