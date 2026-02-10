import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import classification_report, accuracy_score

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")

def load_and_join_data():
    print("Loading datasets for testing (TF-IDF + SVM)...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path)
        # Evaluation on a sample for speed
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
    print("Preprocessing test data...")
    text_cols = ['text', 'name', 'categories', 'name_user', 'city', 'state']
    for col in text_cols:
        df[col] = df[col].astype(str).replace('nan', '')
    
    combined_text = (df['text'] + " " + df['name'] + " " + 
                     df['categories'] + " " + df['name_user'] + " " +
                     df['city'] + " " + df['state'])
    y = df['stars'].values.astype(int)
    
    return combined_text, y

def test():
    # 1. Load resources
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Saved model or vectorizer not found. Please run train_model.py in this folder first.")
        sys.exit(1)
        
    print("Loading vectorizer...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    print("Loading SVM model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load and process data
    df = load_and_join_data()
    text_data, y_true = preprocess_data(df)
    
    # 3. Vectorization
    print("Vectorizing test data (TF-IDF)...")
    X_test = vectorizer.transform(text_data)
    
    # 4. Inference
    print("Running inference...")
    y_pred = model.predict(X_test)
            
    # 5. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    target_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    test()
