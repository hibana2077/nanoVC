import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from pprint import pprint
from elevenlabs import ElevenLabs

# Get the ElevenLabs API key from the environment variable
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", 'sk_953622cd2e2545d6c90793a7abb5809fba5ec1358954efbc')
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Voice ID list
VOICE_ID_LIST = [
    "IKne3meq5aSn9XLyUdCD",  # English (AU)
    "cgSgspJ2msm6clMCkdW9",  # English (US)
    "XB0fDUnXU5powFXDhCwa",  # English (Swedish)
    "N2lVS1w4EtoT3dr4eOWO",  # English (Transatlantic)
    "pFZP5JQG7iQjIQuC4Bku",  # English (UK)
]

def process_audio_data(text: str, voice_id: str = "cgSgspJ2msm6clMCkdW9") -> np.ndarray:
    """
    Transform text to speech using the ElevenLabs API.

    Args:
        text (str): The text to convert to speech.
        voice_id (str): The voice ID to use for the conversion.

    Returns:
        np.ndarray: The audio data as a NumPy array.
    """
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="pcm_22050",
        text=text,
        model_id="eleven_multilingual_v2",
    )

    # 將數據轉為 NumPy 陣列
    audio_data = b"".join(response)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    return audio_array

def generate_tts_pairs(voice_list, sentences, output_file="tts_dataset.npz"):
    """
    According to the voice list and sentences, generate the TTS dataset and save it as a .npz file.

    Args:
        voice_list (list): The list of voice IDs.
        sentences (list): The list of sentences.
        output_file (str): The output file name.
    """
    num_voices = len(voice_list)
    num_sentences = len(sentences)
    data = {
        "inputs": [],
        "ground_truths": []
    }

    data_array = []
    idx_array = []
    for v_idx, num_voice in tqdm(enumerate(voice_list), desc="Processing voices", total=len(voice_list)):
        voice_array = []
        voice_idx_array = []
        for s_idx, num_sentence in tqdm(enumerate(sentences), desc=f"Voice {v_idx+1} sentences", total=len(sentences), leave=False):
            audio_data = process_audio_data(num_sentence, num_voice)
            voice_array.append(audio_data)
            voice_idx_array.append((v_idx, s_idx))
        data_array.append(voice_array)
        idx_array.append(voice_idx_array)

    pprint(idx_array)
    # random sampling two voices save to wav
    random_voices = random.sample(range(num_voices), 2)
    for i in random_voices:
        for j in range(num_sentences):
            write(f"voice_{i+1}_sentence_{j+1}.wav", 22050, data_array[i][j])

    # making pairs

    for i in range(num_voices):
        for j in range(num_sentences):
            for k in range(num_voices):
                if k != i:  # V1 and Vn are not the same
                    for l in range(num_sentences):
                        if l != j:  # S1 and Sn are not the same
                            data["inputs"].append((data_array[i][j], data_array[k][l]))
                            data["ground_truths"].append(data_array[k][j])

    np.savez(output_file, 
             inputs=np.array(data["inputs"], dtype=object),
             ground_truths=np.array(data["ground_truths"], dtype=object))
    

def calculate_total_pairs(voices, sentences):
    """
    Calculate the total number of possible pairs.

    Args:
        voices (int): The number of voices.
        sentences (int): The number of sentences.

    Returns:
        int: The total number of possible pairs.
    """
    return voices * sentences * (voices - 1) * (sentences - 1)

def calculate_max_tts_generation_time(voices, sentences):
    """
    Calculate the maximum time required to generate all TTS pairs.
    """
    return voices * sentences

# Execute the following code when running this script
if __name__ == "__main__":
    # example sentences
    sentences = ["The first move is what sets everything in motion.", "Don't gentle into that good night"]
    # sentences = pd.read_csv('./DNGGITGN.csv')['english'].tolist()

    # calculate the total number of possible pairs
    total_pairs = calculate_total_pairs(len(VOICE_ID_LIST), len(sentences))
    print(f"Total possible pairs: {total_pairs}")
    max_time = calculate_max_tts_generation_time(len(VOICE_ID_LIST), len(sentences))
    print(f"Maximum time required to generate all TTS pairs: {max_time} times")

    # save the TTS dataset
    generate_tts_pairs(VOICE_ID_LIST, sentences, output_file="tts_dataset_test.npz")
    print("TTS dataset saved successfully!")
