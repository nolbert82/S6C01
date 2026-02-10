import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")

MAX_FEATURES = 2500  # Limited to avoid memory issues with BoW
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

def load_and_join_data():
    print("Loading datasets...")
    # Paths to the CSV files
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    # Load CSVs
    try:
        df_business = pd.read_csv(business_path)
        df_reviews = pd.read_csv(review_path)
        df_users = pd.read_csv(user_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        sys.exit(1)

    print("Joining datasets...")
    # Join reviews with business and user data
    # Business join - include city and state
    df_merged = pd.merge(df_reviews, df_business[['business_id', 'name', 'categories', 'stars', 'city', 'state']], 
                         on='business_id', how='left', suffixes=('', '_business'))
    
    # User join
    df_merged = pd.merge(df_merged, df_users[['user_id', 'name', 'review_count', 'average_stars']], 
                         on='user_id', how='left', suffixes=('', '_user'))
    
    return df_merged

def preprocess_data(df):
    print("Preprocessing text and features...")
    
    # Combine all text data as requested: review text, business name, categories, user name
    # Handle NaNs
    df['text'] = df['text'].fillna('')
    df['name'] = df['name'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['name_user'] = df['name_user'].fillna('')
    
    # Combined text for BoW
    # Including city and state as they are also text data
    df['city'] = df['city'].fillna('')
    df['state'] = df['state'].fillna('')
    
    combined_text = (df['text'] + " " + df['name'] + " " + 
                     df['categories'] + " " + df['name_user'] + " " +
                     df['city'] + " " + df['state'])
    
    # Target variable (stars 1-5 -> 0-4 for classification)
    y = df['stars'].values.astype(int) - 1
    
    # Ensure stars are within bounds [0, 4]
    y = np.clip(y, 0, 4)
    
    return combined_text, y

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    df = load_and_join_data()
    
    # 2. Preprocess
    text_data, y = preprocess_data(df)
    
    # 3. Vectorization (BoW)
    print(f"Vectorizing text (BoW, {MAX_FEATURES} features)...")
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words='english')
    X_bow = vectorizer.fit_transform(text_data).toarray()
    
    # 4. Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X_bow, y, test_size=0.2)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # 5. DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 6. Model initialization
    input_dim = MAX_FEATURES
    output_dim = 5  # 1 to 5 stars
    model = LinearModel(input_dim, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    # 8. Save model and vectorizer
    print(f"Saving model to {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
    # Save vectorizer for inference
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save model metadata (like input_dim) to a simple file if needed
    with open(os.path.join(MODEL_DIR, "config.txt"), "w") as f:
        f.write(f"input_dim={input_dim}")

    print("Training complete.")

if __name__ == "__main__":
    train()
