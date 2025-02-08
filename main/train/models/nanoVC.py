import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from xcodec2.modeling_xcodec2 import XCodec2Model
from .modules import DeepseekV3MLP

class NanoVC(nn.Module):
    def __init__(self, training: bool = False):
        super(NanoVC, self).__init__()
        self.Training = training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 載入 xcodec2 模型並設定為 eval 模式
        self.xcodec2 = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").eval().to(self.device)
        self.mlp = DeepseekV3MLP(hidden_size=2048, intermediate_size=2048, hidden_act="gelu_fast").to(self.device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        參數:
            x1: 用於提取語義特徵的語音訊號張量，形狀 [B, length]
            x2: 用於提取 VQ 嵌入的語音訊號張量，形狀 [B, length]
            sample_rate: 取樣率 (預設為 16000)
        回傳:
            recon_audio: 經過語音轉換後的重建語音張量
        """
        # 計算需要填充的長度，使得長度為 320 的倍數
        pad_x1 = (320 - (x1.shape[1] % 320)) % 320
        pad_x2 = (320 - (x2.shape[1] % 320)) % 320

        # 對 x1 與 x2 進行填充
        x1_padded = F.pad(x1, (0, pad_x1))
        x2_padded = F.pad(x2, (0, pad_x2))

        with torch.no_grad():
            # === 處理 x1 的語義特徵 (支援 batch 推論) ===
            input_features_list = []
            # 逐筆處理 x1 中的每個樣本
            for i in range(x1_padded.shape[0]):
                # 將單筆音訊轉至 CPU 處理（因模型限制）
                audio = x1_padded[i].cpu()
                # 為避免邊緣效應，額外於前後填充 160 個樣本
                audio = F.pad(audio, (160, 160))
                # 呼叫 feature_extractor 處理單筆資料，返回 shape: [1, frames, feat_dim]
                features = self.xcodec2.feature_extractor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features
                input_features_list.append(features)
            # 合併所有單筆特徵成 batch，得到 shape: [B, frames, feat_dim]
            input_features = torch.cat(input_features_list, dim=0).to(self.device)

            semantic_output = self.xcodec2.semantic_model(input_features)
            # 取出第 16 層的隱藏狀態，並轉置為 [B, channels, frames]
            semantic_hidden = semantic_output.hidden_states[16].transpose(1, 2)
            semantic_encoded = self.xcodec2.SemanticEncoder_module(semantic_hidden)

            # === 處理 x2 的 VQ 嵌入 ===
            x2_device = x2_padded.to(self.device)
            # 為 x2 新增 channel 維度後進行編碼，再轉置得到 shape: [B, channels, frames]
            vq_emb = self.xcodec2.CodecEnc(x2_device.unsqueeze(1)).transpose(1, 2)

        # === 拼接特徵並進行後續處理 ===
        # 沿 channel 維度拼接語義特徵與 VQ 嵌入，得到 shape: [B, 2048, frames]
        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)
        # 轉置為 [B, frames, channels] 以符合 MLP 輸入格式
        concat_emb = self.mlp(concat_emb.transpose(1, 2))
        # 經過 fc_prior 層後再轉置回 [B, channels, frames]
        concat_emb = self.xcodec2.fc_prior(concat_emb).transpose(1, 2)

        _, vq_code, _ = self.xcodec2.generator(concat_emb, vq=True)    
        vq_post_emb = self.xcodec2.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

        # 7) fc_post_a
        vq_post_emb = self.xcodec2.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

        # 8) 最后解码成波形
        recon_audio = self.xcodec2.generator(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]

        # 截斷填充部分，使輸出語音長度與 x2 原始長度一致
        recon_audio = recon_audio[:, :, :x2.shape[1]].squeeze(1)
        return recon_audio

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = NanoVC().to(device)
    X1 = torch.rand(2, 32000).to(device)
    X2 = torch.rand(2, 32000).to(device)
    ts = time.time()
    output = model(X1, X2)
    te = time.time()
    print(f"Output shape: {output.shape}, Time taken: {te-ts:.2f} seconds")

    # Fvcore Flops
    # fva = FlopCountAnalysis(model, (X1, X2))
    # print(f"Flops: {fva.total()/1e9:.2f} GFLOPs")

    # params = parameter_count(model)
    print(parameter_count_table(model))