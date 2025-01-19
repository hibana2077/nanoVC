from elevenlabs import ElevenLabs
import pandas as pd

client = ElevenLabs(
    api_key="YOUR_API_KEY",
)
client.text_to_speech.convert_as_stream(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    output_format="pcm_22050",
    text="The first move is what sets everything in motion.",
    model_id="eleven_multilingual_v2",
)