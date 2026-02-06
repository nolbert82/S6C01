import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import scipy.sparse

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'linear_regression', 'saved_model')

def load_data():
    """Loads and merges the datasets (Same as train_model)."""
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

def prepare_features(df, vectorizer, scaler):
    """Prepares features using loaded artifacts."""
    print("Preparing features...")
    
    # 1. Target
    if 'stars_review' in df.columns:
        y = df['stars_review']
    elif 'stars' in df.columns:
        y = df['stars']
    else:
        y = None

    # 2. Text Features
    # MUST MATCH train_model logic exactly
    text_cols = []
    candidates = [
        'text', 
        'name_business', 'address', 'city', 'state', 'categories', 
        'name_user', 'name', 
        'attributes', 'hours'
    ]
    for col in candidates:
        if col in df.columns:
            text_cols.append(col)
            
    print(f"Combining text from columns: {text_cols}")
    text_data = df[text_cols].astype(str).fillna('').agg(' '.join, axis=1)
    
    print("Transforming text...")
    X_text = vectorizer.transform(text_data)
        
    # 3. Numeric Features
    num_candidates = [
        'useful_review', 'funny_review', 'cool_review',
        'stars_business', 'review_count_business', 'latitude', 'longitude', 'is_open',
        'average_stars', 'review_count_user', 'useful_user', 'funny_user', 'cool_user', 'fans'
    ]
    
    valid_num_cols = [c for c in num_candidates if c in df.columns]
    
    if 'useful' in df.columns and 'useful_review' not in df.columns:
        valid_num_cols.append('useful')
    if 'funny' in df.columns and 'funny_review' not in df.columns:
        valid_num_cols.append('funny')
    
    print(f"Using numeric features: {valid_num_cols}")
    df_num = df[valid_num_cols].fillna(0)
    
    print("Transforming numeric features...")
    X_num = scaler.transform(df_num)
        
    X = scipy.sparse.hstack((X_text, X_num))
    
    return X, y

def test_model():
    # Load Artifacts
    print("Loading model artifacts...")
    try:
        with open(os.path.join(MODEL_DIR, 'linear_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Model artifacts not found. Please run train_model.py first.")
        return

    # Load Data
    df = load_data()
    
    # Split to get the same test set
    print("Splitting data to isolate test set...")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare Features
    X_test, y_test = prepare_features(test_df, vectorizer, scaler)
    
    # Predict
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("Evaluation Results:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Classification metrics
    y_pred_rounded = np.round(y_pred)
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    accuracy = accuracy_score(y_test, y_pred_rounded)
    print(f"Accuracy (rounded to 1-5): {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rounded))

if __name__ == "__main__":
    test_model()
