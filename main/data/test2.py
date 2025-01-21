import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.io.wavfile import write, read
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, 
                 npz_path, 
                 sample_rate=22050, 
                 max_seconds=10, 
                 transform=None):
        """
        Args:
            npz_path (str): 讀取 .npz 檔案的路徑。
            sample_rate (int): 音訊的採樣率 (預設 22050)。
            max_seconds (int): 最長秒數，用於統一 padding 長度 (預設 10 秒)。
            transform (callable, optional): 若有需要可在取資料時進行的資料轉換函式。
        """
        super(AudioDataset, self).__init__()
        
        # 載入 npz 檔案
        data = np.load(npz_path, allow_pickle=True)
        
        # 'inputs' 為二維陣列，每一筆資料包含了 (input1, input2)
        self.inputs = data["inputs"]
        # 'ground_truths' 為一維陣列，每一筆為 ground_truth 的音訊序列
        self.ground_truths = data["ground_truths"]
        
        # 可以自定義的 transform，若需要在此做特徵轉換等預處理
        self.transform = transform
        
        # 預先計算需要的最大樣本數：10 秒 * 22050 = 220500
        self.max_samples = sample_rate * max_seconds

        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self):
        """回傳資料集長度。"""
        return len(self.inputs)

    def __getitem__(self, idx):
        """取得第 idx 筆資料，轉成固定長度的 PyTorch tensor。"""
        # 取出音訊對 (input1, input2)
        input1, input2 = self.inputs[idx]
        ground_truth = self.ground_truths[idx]

        print(input1.shape)
        write("input1.wav", 22050, input1)
        print(input2.shape)
        write("input2.wav", 22050, input2)
        print(ground_truth.shape)
        write("ground_truth.wav", 22050, ground_truth)

        # 將 numpy array 轉成 PyTorch float tensor
        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        ground_truth = torch.from_numpy(ground_truth).float()

        # 進行 padding 或截斷處理
        input1 = self.pad_or_truncate(input1, self.max_samples)
        input2 = self.pad_or_truncate(input2, self.max_samples)
        ground_truth = self.pad_or_truncate(ground_truth, self.max_samples)

        # 若有自定義 transform，這裡可以進一步處理
        if self.transform:
            input1 = self.transform(input1)
            input2 = self.transform(input2)
            ground_truth = self.transform(ground_truth)

        # 回傳 ((input1, input2), ground_truth)
        return (input1, input2), ground_truth

    def pad_or_truncate(self, wave_tensor, max_samples):
        """若音訊長度 > max_samples，進行截斷；若小於 max_samples，則右側 zero-padding。"""
        current_length = wave_tensor.shape[0]
        print(current_length)

        if current_length > max_samples:
            # 截斷
            wave_tensor = wave_tensor[:max_samples]
        else:
            # padding 到 max_samples
            # 可用 torch.nn.functional.pad 也可用 concat
            pad_length = max_samples - current_length
            # wave_tensor shape [current_length]
            # F.pad 的 padding 參數: (left, right)
            wave_tensor = F.pad(wave_tensor, (0, pad_length), "constant", 0.0)
        
        wave_tensor = wave_tensor.reshape(self.max_seconds, self.sample_rate)

        return wave_tensor

# 假設你的 .npz 檔案在 ./tts_dataset.npz
dataset = AudioDataset(
    npz_path="./tts_dataset.npz", 
    sample_rate=22050, 
    max_seconds=10
)

# 取得第 0 筆資料
(inputs, ground_truth) = dataset[0]
print(f"inputs[0] shape: {inputs[0].shape}")
print(f"inputs[1] shape: {inputs[1].shape}")
print(f"ground_truth shape: {ground_truth.shape}")