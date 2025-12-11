import os
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"
LOCAL_MODEL_DIR = "models/bert-base-uncased"

print(f"Downloading {MODEL_NAME} to {LOCAL_MODEL_DIR}...")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Download tokenizer and model
print("Downloading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)

print("Downloading model...")
model = BertModel.from_pretrained(MODEL_NAME)
model.save_pretrained(LOCAL_MODEL_DIR)

print(f"Successfully downloaded {MODEL_NAME} to {LOCAL_MODEL_DIR}")
