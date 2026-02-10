import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "csv"))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")

# Using more features since TF-IDF is more informative and SVM handles high-dim well
MAX_FEATURES = 10000 

def load_and_join_data():
    print("Loading datasets for TF-IDF + SVM model...")
    business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.csv")
    review_path = os.path.join(DATA_DIR, "yelp_academic_reviews4students.csv")
    user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user4students.csv")

    try:
        df_business = pd.read_csv(business_path)
        # We can handle more data with LinearSVC as it's efficient, but let's keep it reasonable
        df_reviews = pd.read_csv(review_path)
        if len(df_reviews) > 150000:
            print("Sampling 150k reviews for training...")
            df_reviews = df_reviews.sample(n=150000, random_state=42)
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
    print("Preprocessing text...")
    # Clean and combine text fields
    text_cols = ['text', 'name', 'categories', 'name_user', 'city', 'state']
    for col in text_cols:
        df[col] = df[col].astype(str).replace('nan', '')
    
    combined_text = (df['text'] + " " + df['name'] + " " + 
                     df['categories'] + " " + df['name_user'] + " " +
                     df['city'] + " " + df['state'])
    
    y = df['stars'].values.astype(int)
    return combined_text, y

def train():
    # Note: Scikit-learn LinearSVC runs on CPU. It is extremely fast for sparse TF-IDF data.
    print("Algorithm: Linear SVM with TF-IDF")

    # 1. Load data
    df = load_and_join_data()
    
    # 2. Preprocess
    text_data, y = preprocess_data(df)
    
    # 3. Train/Test split
    X_train_text, X_val_text, y_train, y_val = train_test_split(text_data, y, test_size=0.2, random_state=42)
    
    # 4. TF-IDF Vectorization
    print(f"Vectorizing text (TF-IDF, {MAX_FEATURES} features)...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    
    # 5. Model initialization & Training
    print("Training LinearSVC...")
    # LinearSVC is generally better than SVC for text (linearly separable in high-dim)
    model = LinearSVC(C=1.0, max_iter=1000, dual='auto')
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    print("Evaluating...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # 7. Save model and vectorizer
    print(f"Saving artifacts to {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train()
