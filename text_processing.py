import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Download NLTK resources (Run only once)
nltk.download('punkt')
nltk.download('stopwords')
with open("dataset/rawtext.txt", "r", encoding="utf-8") as file:
    text = file.read()
    
print("Original Text:\n")
print(text)

text = text.lower()
text = re.sub(r"\d+", "", text)
text = re.sub(r"[^\w\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()

print("\nCleaned Text:\n")
print(text)

words = word_tokenize(text)

print("\nWord Tokens:")
print(words)

pd.DataFrame({"Clean_Text": [text]}).to_csv(
    "dataset/clean_text.csv",
    index=False
)

pd.DataFrame({"Token": words}).to_csv(
    "dataset/tokens.csv",
    index=False
)

print("\nFiles saved successfully.")
