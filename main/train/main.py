import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from models.nanoVC import NanoVC

class AudioDataset(Dataset):
    def __init__(self, 
                 npz_path, 
                 sample_rate=16000, 
                 max_seconds=10, 
                 transform=None):
        """
        Args:
            npz_path (str): 讀取 .npz 檔案的路徑。
            sample_rate (int): 音訊的採樣率 (預設 16000)。
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
        
        # 預先計算需要的最大樣本數：10 秒 * 16000 = 160000
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

        # 將 numpy array 轉成 PyTorch float tensor
        input1 = torch.from_numpy(input1).float()/32768
        input2 = torch.from_numpy(input2).float()/32768
        ground_truth = torch.from_numpy(ground_truth).float()/32768

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

        if current_length > max_samples:
            # 截斷
            wave_tensor = wave_tensor[:max_samples]
        else:
            # padding 到 max_samples
            # 可用 torch.nn.functional.pad 也可用 concat
            pad_length = max_samples - current_length
            # wave_tensor shape [current_length]
            # F.pad 的 padding 參數: (left, right)
            wave_tensor = F.pad(wave_tensor, (0, pad_length), "constant", 1e-7)
        
        wave_tensor = wave_tensor.reshape(self.max_seconds, self.sample_rate)

        return wave_tensor

# 假設你的 .npz 檔案在 ./tts_dataset.npz
dataset = AudioDataset(
    npz_path="../data/tts_dataset.npz", 
    sample_rate=16000, 
    max_seconds=3
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset,
                            batch_size=8,
                            shuffle=True)
val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            shuffle=False)


# 準備模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = NanoVC(Training=True).to(device)

# 準備損失函數和優化器
# criteria_a = nn.MSELoss()
criteria_b = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1).mean()
# optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=0.0003)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0003)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def create_stft_mel_scales(device, sample_rate=16000, n_mels=80):
    n_ffts = [512, 1024, 2048]
    stfts = [
        torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=n_fft // 4,
            power=None
        ).to(device)
        for n_fft in n_ffts
    ]

    mel_scales = [
        torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=(n_fft // 2) + 1
        ).to(device)
        for n_fft in n_ffts
    ]

    return stfts, mel_scales

stfts, mel_scales = create_stft_mel_scales(device)

def stft_mel_loss(output_wave, gt_wave):
    losses = []

    for stft, mel_scale in zip(stfts, mel_scales):
        stft_out = stft(output_wave)  # shape: (batch, freq, time)
        stft_gt = stft(gt_wave)

        mag_out = stft_out.abs()  # 取 magnitude
        mag_gt = stft_gt.abs()

        mel_out = mel_scale(mag_out)  # shape: (batch, n_mels, time)
        mel_gt = mel_scale(mag_gt)

        # 加 log
        mel_out_log = torch.log(mel_out + 1e-10)
        mel_gt_log = torch.log(mel_gt + 1e-10)

        losses.append(F.l1_loss(mel_out_log, mel_gt_log))

    # 加權求和 (權重可以根據需求調整)
    total_loss = sum(losses) / len(losses)
    return total_loss

# Test criterion
for i, ((input1, input2), gt) in enumerate(train_loader):
    input1, input2, gt = input1.to(device), input2.to(device), gt.to(device)
    output, output_f, input2_f = model(input1, input2)
    print(output.shape, output_f[-1].shape, input2_f[-1].shape, gt.shape)
    print(stft_mel_loss(output, gt), criteria_b(output_f[-1], input2_f[-1])) # , criteria_b(input2_f, output_f)
    break

# 訓練與測試模型
best_model = None
for epoch in range(200):
    # 訓練階段
    model.train()
    model.Training = True
    running_sftf_loss = 0.0
    running_cs_loss = 0.0
    running_l1_loss = 0.0
    running_loss = 0.0
    running_norm = 0

    best_loss = 1e9

    for i, ((input1, input2), gt) in enumerate(tqdm(train_loader)):
        input1, input2, gt = input1.to(device), input2.to(device), gt.to(device)
        optimizer.zero_grad()
        output, output_f, input2_f = model(input1, input2)
        # output_f = output_f[-1]
        # input2_f = input2_f[-1]
        stft_loss = stft_mel_loss(output, gt)
        l1_loss = F.l1_loss(output, gt)
        # cs_loss = criteria_b(output_f, input2_f)
        loss = stft_loss + l1_loss# + cs_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # running_cs_loss += cs_loss.item()
        running_sftf_loss += stft_loss.item()
        running_l1_loss += l1_loss.item()
        running_loss += loss.item()
        running_norm += total_norm
    print(f"Epoch {epoch+1}, Training Loss: {running_loss/(i+1):.5f}, STFT Loss: {running_sftf_loss/(i+1):.5f}, L1 Loss: {running_l1_loss/(i+1):.5f}, Cosine Similarity Loss: {running_cs_loss/(i+1):.5f}, Norm: {running_norm/(i+1):.5f}")
    scheduler.step()

    # 驗證階段
    model.eval()
    model.Training = False
    val_loss = 0.0
    with torch.no_grad():
        for i, ((input1, input2), gt) in enumerate(tqdm(val_loader)):
            input1, input2, gt = input1.to(device), input2.to(device), gt.to(device)
            # output, _, _ = model(input1, input2)
            output = model(input1, input2)
            loss = stft_mel_loss(output, gt)
            val_loss += loss.item()
    if val_loss/(i+1) < best_loss:
        best_loss = val_loss/(i+1)
        best_model = model.state_dict()
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/(i+1):.5f}")

# 保存最好的模型
torch.save(best_model, "best_model.pth")
