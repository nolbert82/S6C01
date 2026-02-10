import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import classes and functions from train_model
# Adjust sys.path if necessary, but since they are in the same dir...
try:
    from train_model import MLP, YelpDataset, load_data, prepare_features
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_model import MLP, YelpDataset, load_data, prepare_features

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'tfidf_mlp', 'saved_model')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def test_model():
    print("Loading artifacts...")
    try:
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            
        input_dim = config['input_dim']
        
        model = MLP(input_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mlp_model.pth'), map_location=device))
        model.eval()
        
    except FileNotFoundError as e:
        print(f"Artifacts not found: {e}")
        print("Please run train_model.py first.")
        return

    # Load Data
    df = load_data()
    
    print("Splitting data to isolate test set...")
    # Must use same seed as train_model
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare Features
    # Note: prepare_features returns X_text, X_num, y, vectorizer, scaler
    X_test_text, X_test_num, y_test, _, _ = prepare_features(test_df, vectorizer=vectorizer, scaler=scaler, is_training=False)
    
    # Dataset and DataLoader
    print("Creating DataLoader...")
    test_dataset = YelpDataset(X_test_text, X_test_num, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Prediction Loop
    print("Predicting...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            
            # Helper to handle single-element batches which return 0-d tensors
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Evaluation
    print("\nEvaluation Results:")
    mse = mean_squared_error(all_targets, all_preds)
    print(f"Mean Squared Error: {mse:.4f}")
    
    y_pred_rounded = np.round(all_preds).clip(1, 5)
    accuracy = accuracy_score(all_targets, y_pred_rounded)
    print(f"Accuracy (rounded to 1-5): {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, y_pred_rounded))

if __name__ == "__main__":
    test_model()
