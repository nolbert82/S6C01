import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report, accuracy_score
import os
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16

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
        embed_dim = self.bert.config.hidden_size
        
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(3 * 128, n_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = distilbert_output.last_hidden_state
        x = x.permute(0, 2, 1)
        
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)
        
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        
        combined = torch.cat((x1, x2, x3), dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)

def load_and_join_data():
    print("Loading datasets for testing (BERT + CNN)...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path, low_memory=False)
        # Test on a small sample for feasibility
        df_reviews = pd.read_csv(review_path, low_memory=False).sample(n=2000, random_state=42)
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

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load resources
    model_path = os.path.join(MODEL_DIR, "model.pth")
    if not os.path.exists(model_path):
        print("Error: Saved model not found. Please run train_model.py first.")
        sys.exit(1)
        
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    
    print("Loading model...")
    model = BertCNN(MODEL_NAME, 5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load data
    df = load_and_join_data()
    texts, labels = preprocess_data(df)
    test_dataset = YelpDataset(texts, labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. Inference
    print("Running inference...")
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    # 4. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    target_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    test()
