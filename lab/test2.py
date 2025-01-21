import torch
import torch.nn.functional as F

t4d = torch.tensor([1,2,3,4,5])
p1d = (0, 5)
out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
print(out.size())
print(out)
out = out.reshape(2, 5)
print(out.size())
print(out)