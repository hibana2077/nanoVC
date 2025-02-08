import torch
import soundfile as sf
from transformers import AutoConfig

 
from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUSTAudio/xcodec2"

model = XCodec2Model.from_pretrained(model_path)
model.eval().cuda()

 
wav, sr = sf.read("./voice_2_sentence_1.wav")  # Shape: (T, )
print("Original:", wav.shape, sr)
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)
print("Input:", wav_tensor.shape, "Range:", wav_tensor.min(), wav_tensor.max())

 
with torch.no_grad():
   # Only 16khz speech
   # Only supports single input. For batch inference, please refer to the link below.
    vq_code = model.encode_code(input_waveform=wav_tensor)
    print("Code:", vq_code.shape)

    recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

 
sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
print("Done! Check reconstructed.wav")
print(model.feature_extractor)