import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer
model_path = "finetuned_model"
tokenizer_path = "finetuned_tokenizer"

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