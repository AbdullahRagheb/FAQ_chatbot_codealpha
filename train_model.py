import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

# Define the FAQDataset class
class FAQDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        question = self.df.iloc[idx, 0]
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Squeeze to remove extra dimension
        inputs['labels'] = torch.tensor(idx)  # Using the index as the label
        return inputs

# Load the dataset
dataset = FAQDataset('faqs.csv')

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Load the pre-trained model and move it to the device
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset))
model.to(device)

# Define the training loop
def train(dataloader, model, optimizer):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Create DataLoader
train_dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

# Train the model
for epoch in range(training_args.num_train_epochs):
    train(train_dataloader, model, optimizer)

# Save the trained model and tokenizer
model.save_pretrained('./model')
BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('./model')
