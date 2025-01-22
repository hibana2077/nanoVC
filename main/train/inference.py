import numpy as np
import torch
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
from models.nanoVC import NanoVC
from torch.utils.data import Dataset
from fvcore.nn import FlopCountAnalysis, parameter_count_table


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
            wave_tensor = F.pad(wave_tensor, (0, pad_length), "constant", 0.0)
        
        wave_tensor = wave_tensor.reshape(self.max_seconds, self.sample_rate)

        return wave_tensor

def load_model(model_path, device):
    model = NanoVC(Training=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def infer_and_save_to_wav(model, input_tensor, output_path, sample_rate=22050):
    with torch.no_grad():
        # 假設 input_tensor 是準備好的輸入數據
        output = model(*input_tensor)
        output_wave = output.view(-1).cpu().squeeze().numpy()
        print(max(output_wave), min(output_wave))
        print(f"Output shape: {output_wave.shape}") # Output shape: (220500,)
        # 保存為 .wav
        wavfile.write(output_path, sample_rate, output_wave)
        wavfile.write("10times_output.wav", sample_rate, output_wave*10)

if __name__ == "__main__":
    
    dataset = AudioDataset(
        npz_path="../data/tts_dataset.npz",
        sample_rate=16000, 
        max_seconds=3
    )
    data = dataset[0]
    sample_input1, sample_input2 = data[0]
    # 保存兩個輸入音訊
    wavfile.write("input1.wav", 16000, sample_input1.squeeze().view(-1).numpy())
    print(f"Input1 saved to input1.wav")
    wavfile.write("input2.wav", 16000, sample_input2.squeeze().view(-1).numpy())
    print(f"Input2 saved to input2.wav")
    sample_input1 = sample_input1.unsqueeze(0)
    sample_input2 = sample_input2.unsqueeze(0)
    print(f"Input1 shape: {sample_input1.shape}, Input2 shape: {sample_input2.shape}")
    gt = data[1]
    # 保存 ground truth
    wavfile.write("ground_truth.wav", 16000, gt.squeeze().view(-1).numpy())
    print(f"Ground truth saved to ground_truth.wav")
    print(f"Ground truth shape: {gt.shape}")
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入模型
    model_path = "./model.pth"
    model = load_model(model_path, device)
    input_tensor = (sample_input1.to(device), sample_input2.to(device))
    # print model size
    print(parameter_count_table(model))
    # print model flops
    fva = FlopCountAnalysis(model, input_tensor)
    if fva.total() > 1e9:
        print(f"Flops: {fva.total()/1e9:.2f} GFLOPs")
    elif fva.total() > 1e6:
        print(f"Flops: {fva.total()/1e6:.2f} MFLOPs")
    else:
        print(f"Flops: {fva.total()} FLOPs")

    # 推論並保存結果
    output_path = "output.wav"
    infer_and_save_to_wav(model, input_tensor, output_path)
    print(f"Inference result saved to {output_path}")
