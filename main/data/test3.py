import numpy as np
from scipy.io.wavfile import write

# output_file = './tts_dataset_test.npz'
output_file = './tts_dataset.npz'
data = np.load(output_file, allow_pickle=True)
inputs = data["inputs"]   # shape = [N], 裏面每個元素是 (array, array)
gts    = data["ground_truths"]

# 隨機抽一筆
(inp1, inp2) = inputs[0]
gt = gts[0]

print(inp1.dtype, inp1.shape, max(inp1), min(inp1))
print(inp2.dtype, inp2.shape, max(inp2), min(inp2))
print(gt.dtype, gt.shape, max(gt), min(gt))

write("inp1.wav", 22050, inp1)
write("inp2.wav", 22050, inp2)
write("gt.wav" , 22050, gt)
