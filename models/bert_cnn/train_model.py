import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
import os
import sys
import pickle

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")

# BERT/CNN Params
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16 # Small batch size for BERT
EPOCHS = 10
LEARNING_RATE = 2e-5 # Fine-tuning learning rate

class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertCNN(nn.Module):
    def __init__(self, model_name, n_classes):
        super(BertCNN, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freezing BERT is often faster and better if only training the CNN part, 
        # but let's keep it trainable for "fine-tuning" as per usual BERT practices.
        # Alternatively, you can freeze it by: 
        # for param in self.bert.parameters(): param.requires_grad = False
        
        embed_dim = self.bert.config.hidden_size
        
        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(3 * 128, n_classes)

    def forward(self, input_ids, attention_mask):
        # DistilBERT output: (batch_size, seq_len, hidden_size)
        distilbert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = distilbert_output.last_hidden_state # [B, L, H]
        
        # Prepare for Conv1d: (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1) # [B, H, L]
        
        # Convolutions & Pooling
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)
        
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        
        # Concatenate
        combined = torch.cat((x1, x2, x3), dim=1) # [B, 3*128]
        combined = self.dropout(combined)
        
        return self.fc(combined)

def load_and_join_data():
    print("Loading datasets for BERT + CNN model...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path, low_memory=False)
        df_reviews = pd.read_csv(review_path, low_memory=False)
        # Limit data for BERT as it's slow
        if len(df_reviews) > 20000:
            print("Sampling 20k reviews for BERT training...")
            df_reviews = df_reviews.sample(n=20000, random_state=42)
        df_users = pd.read_csv(user_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        sys.exit(1)

    print("Joining datasets...")
    df_merged = pd.merge(df_reviews, df_business[['business_id', 'name', 'categories', 'city', 'state']], 
                         on='business_id', how='left', suffixes=('', '_business'))
    df_merged = pd.merge(df_merged, df_users[['user_id', 'name']], 
                         on='user_id', how='left', suffixes=('', '_user'))
    
    return df_merged

def preprocess_data(df):
    text_cols = ['text', 'name', 'categories', 'name_user', 'city', 'state']
    for col in text_cols:
        df[col] = df[col].astype(str).replace('nan', '')
    
    combined_text = (df['text'] + " [SEP] " + df['name'] + " " + 
                     df['categories'] + " " + df['name_user'] + " " +
                     df['city'] + " " + df['state'])
    
    y = df['stars'].values.astype(int) - 1
    y = np.clip(y, 0, 4)
    return combined_text.values, y

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    df = load_and_join_data()
    texts, labels = preprocess_data(df)
    
    # 2. Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Datasets
    X_train_t, X_val_t, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    train_dataset = YelpDataset(X_train_t, y_train, tokenizer, MAX_LENGTH)
    val_dataset = YelpDataset(X_val_t, y_val, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 4. Model
    model = BertCNN(MODEL_NAME, 5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print(f"Starting Training ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {100*correct/total:.2f}%")

    # 6. Save Model
    print(f"Saving artifacts to {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
    # Tokenizer is also needed for testing
    tokenizer.save_pretrained(MODEL_DIR)

    print("Training complete.")

if __name__ == "__main__":
    train()
