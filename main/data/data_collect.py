import pandas as pd

url = "https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix/resolve/main/en-af/train-00000-of-00003.parquet?download=true"

df = pd.read_parquet(url)
print(df.head())