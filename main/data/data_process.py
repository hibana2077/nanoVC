import pandas as pd
from tqdm import tqdm
import re

# Read the CSV file
df = pd.read_csv("data.csv")

# Drop "non_english" rows with tqdm progress bar
with tqdm(total=len(df), desc="Cleaning rows", unit="rows") as pbar:
    df = df['english']
    df = df.dropna()
    pbar.update(len(df))

# Additional text cleaning steps
def clean_text(text: str) -> str:
    text = text.strip()  # Remove leading and trailing whitespace
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"[\W_]+", " ", text)  # Remove non-alphanumeric characters
    return text

# Apply cleaning function to the text column
with tqdm(total=len(df), desc="Applying text cleaning", unit="rows") as pbar:
    df = df.apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    pbar.update(len(df))

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_data.csv", index=False)
print("Cleaned data saved to cleaned_data.csv")

# Display the first few rows of the cleaned DataFrame
print(df.head())
