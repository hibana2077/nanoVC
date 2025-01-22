# Let's see how to retrieve time steps for a model
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import torch

# feature extractor
# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h", device="cuda")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h", device="cuda")

test_tensor = torch.rand(1, 3, 16000).to("cuda")

# print(feature_extractor)
print(dir(feature_extractor))

output = feature_extractor(test_tensor, sampling_rate=16000, return_tensors="pt").input_values[0].to("cuda")
print(output)
print(output.shape)
print(test_tensor.device, output.device)
print(torch.equal(test_tensor, output))
# print(feature_extractor.sampling_rate)