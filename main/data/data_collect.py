import pandas as pd
import requests
from tqdm import tqdm

# URL of the Parquet file
url = "https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix/resolve/main/en-af/train-00000-of-00003.parquet?download=true"

def download_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Get total file size
    block_size = 1024  # 1 KB per block

    with open(output_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

# Download the file with progress bar
output_file = "train-00000-of-00003.parquet"
download_with_progress(url, output_file)

# Load the Parquet file into a DataFrame
df = pd.read_parquet(output_file)

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv("data.csv", index=False)
print("Data saved to data.csv")