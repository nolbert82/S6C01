import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import scipy.sparse

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'tfidf_mlp', 'saved_model')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data():
    """Loads and merged the datasets."""
    print("Loading datasets...")
    reviews_path = os.path.join(DATA_DIR, 'yelp_academic_reviews4students.csv')
    business_path = os.path.join(DATA_DIR, 'yelp_academic_dataset_business.csv')
    users_path = os.path.join(DATA_DIR, 'yelp_academic_dataset_user4students.csv')
    
    # Load
    reviews = pd.read_csv(reviews_path)
    business = pd.read_csv(business_path)
    users = pd.read_csv(users_path)
    
    print("Merging datasets...")
    df = pd.merge(reviews, business, on='business_id', how='left', suffixes=('_review', '_business'))
    df = pd.merge(df, users, on='user_id', how='left', suffixes=('', '_user'))
    
    return df

class YelpDataset(Dataset):
    def __init__(self, X_text, X_num, y=None):
        """
        X_text: Sparse matrix (TF-IDF)
        X_num: Numpy array (Scaled numeric features)
        y: Numpy array (Targets)
        """
        self.X_text = X_text
        self.X_num = X_num
        self.y = y
        
    def __len__(self):
        return self.X_text.shape[0]
    
    def __getitem__(self, idx):
        # Convert sparse row to dense for PyTorch
        # Note: Doing this per item to save memory, rather than densifying the whole matrix
        text_vec = torch.tensor(self.X_text[idx].toarray(), dtype=torch.float32).squeeze(0)
        num_vec = torch.tensor(self.X_num[idx], dtype=torch.float32)
        
        features = torch.cat((text_vec, num_vec))
        
        if self.y is not None:
            label = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
            return features, label
        return features

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Regression output
        )
        
    def forward(self, x):
        return self.layers(x)

def prepare_features(df, vectorizer=None, scaler=None, is_training=True):
    print("Preparing features...")
    
    # 1. Target
    y = df['stars_review'] if 'stars_review' in df.columns else None

    # 2. Text Features
    text_cols = [
        'text', 
        'name', 'address', 'city', 'state', 'categories', 'attributes', 'hours',
        'name_user'
    ]
    
    print(f"Combining text from columns: {text_cols}")
    text_data = df[text_cols].astype(str).fillna('').agg(' '.join, axis=1)
    
    # Vectorize
    if is_training:
        print("Fitting TfidfVectorizer...")
        # Limiting features significantly as dense MLP input requires memory
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_text = vectorizer.fit_transform(text_data)
    else:
        print("Transforming text...")
        X_text = vectorizer.transform(text_data)
        
    # 3. Numeric Features
    numeric_cols = [
        'useful', 'funny', 'cool',
        'stars_business', 'review_count', 'latitude', 'longitude', 'is_open',
        'average_stars', 'review_count_user', 'useful_user', 'funny_user', 'cool_user', 'fans'
    ]
    
    print(f"Using numeric features: {numeric_cols}")
    df_num = df[numeric_cols].fillna(0)
    
    # Scale
    if is_training:
        print("Fitting StandardScaler...")
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_num)
    else:
        print("Transforming numeric features...")
        X_num = scaler.transform(df_num)
    
    return X_text, X_num, y, vectorizer, scaler

def train_model():
    df = load_data()
    
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare Features
    X_train_text, X_train_num, y_train, vectorizer, scaler = prepare_features(train_df, is_training=True)
    X_test_text, X_test_num, y_test, _, _ = prepare_features(test_df, vectorizer=vectorizer, scaler=scaler, is_training=False)
    
    # Dataset and DataLoader
    print("Creating DataLoaders...")
    train_dataset = YelpDataset(X_train_text, X_train_num, y_train)
    test_dataset = YelpDataset(X_test_text, X_test_num, y_test)
    
    # Batch size needs to be manageable
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model Setup
    input_dim = X_train_text.shape[1] + X_train_num.shape[1]
    model = MLP(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    print(f"Starting training for {num_epochs} epochs on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(test_loader):.4f}")

    # Evaluation
    print("Final Evaluation...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    mse = mean_squared_error(all_targets, all_preds)
    print(f"MSE: {mse:.4f}")
    
    y_pred_rounded = np.round(all_preds).clip(1, 5)
    accuracy = accuracy_score(all_targets, y_pred_rounded)
    print(f"Accuracy (1-5 stars): {accuracy:.4f}")

    # Save Artifacts
    print(f"Saving artifacts to {MODEL_DIR}...")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'mlp_model.pth'))
    
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save input dimension for reconstruction
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
        pickle.dump({'input_dim': input_dim}, f)

if __name__ == "__main__":
    train_model()
