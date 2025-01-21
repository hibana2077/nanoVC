import numpy as np
from scipy.io.wavfile import write, read

# 設定檔案路徑
input_file = './voice_5_sentence_2.wav'
output_file = './output_voice.wav'

# 讀取 WAV 檔案
sample_rate, data = read(input_file)
print(f"Sample Rate: {sample_rate}")
print(f"Data Type: {data.dtype}")
print(f"Shape: {data.shape}")

# 將資料轉換為 NumPy 陣列（若非陣列）
data_array = np.array(data, dtype=data.dtype)

# 將 NumPy 陣列寫入新的 WAV 檔案
write(output_file, sample_rate, data_array)
print(f"New WAV file saved to {output_file}")
