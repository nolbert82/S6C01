import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import sys
from sklearn.metrics import classification_report, accuracy_score

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")
BATCH_SIZE = 128
MAX_FEATURES = 5000

# We need the same classes defined to load the pickle/model
from train_model import BM25Transformer, MLP

def load_and_join_data():
    print("Loading datasets for testing (BM25 + MLP)...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path)
        df_reviews = pd.read_csv(review_path).sample(n=10000, random_state=42) 
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

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load resources
    model_path = os.path.join(MODEL_DIR, "model.pth")
    transformer_path = os.path.join(MODEL_DIR, "bm25_transformer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(transformer_path):
        print("Error: Saved model or transformer not found. Please run train_model.py in this folder first.")
        sys.exit(1)
        
    print("Loading BM25 transformer...")
    with open(transformer_path, "rb") as f:
        bm25 = pickle.load(f)
        
    print("Loading MLP model...")
    model = MLP(MAX_FEATURES, 5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and process data
    df = load_and_join_data()
    text_data, y_true = preprocess_data(df)
    
    # 3. Vectorization
    print("Vectorizing test data (BM25)...")
    X_test = bm25.transform(text_data).toarray()
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_true)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)
    
    # 4. Inference
    print("Running inference...")
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            
    # 5. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    target_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    test()
