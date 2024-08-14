from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

app = Flask(__name__)

# Load the FAQ data
df = pd.read_csv('faqs.csv')

# Load the trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    inputs = tokenizer(user_question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    
    answer = df.iloc[predicted_index, 1]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

