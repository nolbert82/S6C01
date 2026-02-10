import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")

MAX_FEATURES = 5000 
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class BM25Transformer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words='english')
        self.idf = None
        self.avgdl = None

    def fit(self, texts):
        print("Fitting CountVectorizer...")
        tf = self.vectorizer.fit_transform(texts)
        
        # Calculate doc lengths
        doc_lengths = tf.sum(axis=1).A1
        self.avgdl = np.mean(doc_lengths)
        
        # Calculate IDF
        N = tf.shape[0]
        # Number of docs containing term t
        df = np.diff(tf.tocsc().indptr)
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        
        return self

    def transform(self, texts):
        tf = self.vectorizer.transform(texts)
        doc_lengths = tf.sum(axis=1).A1
        
        # We process in chunks to avoid memory issues if needed, 
        # but for MAX_FEATURES=5000 it should fit.
        tf = tf.tocoo()
        
        # formula: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
        # This is easier to compute on the sparse data directly
        data = tf.data
        rows = tf.row
        cols = tf.col
        
        # BM25 components
        dl_term = self.k1 * (1 - self.b + self.b * (doc_lengths[rows] / self.avgdl))
        new_data = self.idf[cols] * (data * (self.k1 + 1)) / (data + dl_term)
        
        from scipy.sparse import csr_matrix
        return csr_matrix((new_data, (rows, cols)), shape=tf.shape)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def load_and_join_data():
    print("Loading datasets for BM25 + MLP model...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path)
        df_reviews = pd.read_csv(review_path)
        if len(df_reviews) > 100000:
            print("Sampling 100k reviews for training...")
            df_reviews = df_reviews.sample(n=100000, random_state=42)
        df_users = pd.read_csv(user_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        sys.exit(1)

    print("Joining datasets...")
    df_merged = pd.merge(df_reviews, df_business[['business_id', 'name', 'categories', 'stars', 'city', 'state']], 
                         on='business_id', how='left', suffixes=('', '_business'))
    df_merged = pd.merge(df_merged, df_users[['user_id', 'name', 'review_count', 'average_stars']], 
                         on='user_id', how='left', suffixes=('', '_user'))
    
    return df_merged

def preprocess_data(df):
    text_cols = ['text', 'name', 'categories', 'name_user', 'city', 'state']
    for col in text_cols:
        df[col] = df[col].astype(str).replace('nan', '')
    
    combined_text = (df['text'] + " " + df['name'] + " " + 
                     df['categories'] + " " + df['name_user'] + " " +
                     df['city'] + " " + df['state'])
    
    y = df['stars'].values.astype(int) - 1
    y = np.clip(y, 0, 4)
    return combined_text, y

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    df = load_and_join_data()
    text_data, y = preprocess_data(df)
    
    # 2. BM25 Vectorization
    print("Applying BM25 transformation...")
    bm25 = BM25Transformer().fit(text_data)
    X_bm25 = bm25.transform(text_data).toarray()
    
    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(X_bm25, y, test_size=0.15)
    
    # 4. PyTorch setup
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), 
                            batch_size=BATCH_SIZE)
    
    model = MLP(MAX_FEATURES, 5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training
    print("Starting MLP training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {100*correct/len(y_val):.2f}%")

    # 6. Save
    print(f"Saving artifacts to {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
    with open(os.path.join(MODEL_DIR, "bm25_transformer.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train()
