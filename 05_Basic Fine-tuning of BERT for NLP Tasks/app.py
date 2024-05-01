import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

import zipfile
import shutil
import os

def unzip_and_save(zip_file_path, extraction_path):
    # Create the extraction directory if it doesn't exist
    os.makedirs(extraction_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        folder_name = os.path.basename(zip_file_path).split('.')[0]
        zip_ref.extractall(extraction_path)
        source_path = os.path.join(extraction_path, folder_name)
        destination_path = os.path.join(extraction_path, folder_name)
        if os.path.exists(destination_path):
            print(f"Error: Destination path '{destination_path}' already exists")
        else:
            shutil.move(source_path, destination_path)

# Example usage:
# Path to your ZIP file which is your sentimetn analysis model zip
zip_file_path = 'finetuned_bert_sentiment_harsh.zip'  
# Destination folder for extraction
extraction_path = 'bert_model_sentiment_v1'  

unzip_and_save(zip_file_path, extraction_path)

# Load the fine-tuned model and tokenizer
model_path = "bert_model_sentiment_v1/finetuned_bert_sentiment_harsh"
tokenizer_path = "bert_model_sentiment_v1/finetuned_bert_sentiment_harsh"

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_model()

def predict_sentiment(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    tokenized = tokenizer(text, truncation=True, padding=True, return_tensors='pt').to(device)
    outputs = model(**tokenized)

    probs = F.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(outputs.logits, dim=-1).item()
    probs_max = probs.max().detach().cpu().numpy()

    prediction = "Positive" if preds == 1 else "Negative"
    return prediction, probs_max * 100

st.title("Sentiment Analysis App")
text = st.text_area("Enter your text:")

if st.button("Predict Sentiment"):
    if text:
        sentiment, confidence = predict_sentiment(text)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.write("Please enter some text.")