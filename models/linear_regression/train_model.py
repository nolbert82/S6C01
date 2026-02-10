import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import scipy.sparse

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'linear_regression', 'saved_model')

def load_data():
    """Loads and merges the datasets."""
    print("Loading datasets...")
    reviews_path = os.path.join(DATA_DIR, 'yelp_academic_reviews4students.csv')
    business_path = os.path.join(DATA_DIR, 'yelp_academic_dataset_business.csv')
    users_path = os.path.join(DATA_DIR, 'yelp_academic_dataset_user4students.csv')
    
    # Load with low_memory=False to avoid mixed type warnings or specify dtypes
    reviews = pd.read_csv(reviews_path)
    business = pd.read_csv(business_path)
    users = pd.read_csv(users_path)
    
    print("Merging datasets...")
    # Merge Reviews + Business
    # suffixes: _review (default for left), _business (default for right)
    df = pd.merge(reviews, business, on='business_id', how='left', suffixes=('_review', '_business'))
    
    # Merge + Users
    # user_id is in reviews and users
    df = pd.merge(df, users, on='user_id', how='left', suffixes=('', '_user'))
    
    print(f"Full dataset shape: {df.shape}")
    return df

def prepare_features(df, vectorizer=None, scaler=None, is_training=True):
    """Prepares text and numeric features."""
    print("Preparing features...")
    
    # 1. Target
    # Based on merge suffixes ('_review', '_business'), target is 'stars_review'
    y = df['stars_review'] if 'stars_review' in df.columns else None

    # 2. Text Features
    # Explicitly selecting text columns from the merged dataset
    # reviews: text
    # business: name, address, city, state, categories, attributes, hours
    # users: name_user (renamed due to collision with business name)
    text_cols = [
        'text', 
        'name', 'address', 'city', 'state', 'categories', 'attributes', 'hours',
        'name_user'
    ]
    
    print(f"Combining text from columns: {text_cols}")
    
    # Fill NAs and combine
    text_data = df[text_cols].astype(str).fillna('').agg(' '.join, axis=1)
    
    # Vectorize
    if is_training:
        print("Fitting CountVectorizer (Bag of Words)...")
        # Limiting features to avoid memory explosion
        vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        X_text = vectorizer.fit_transform(text_data)
    else:
        print("Transforming text with existing vectorizer...")
        X_text = vectorizer.transform(text_data)
        
    # 3. Numeric Features
    # Explicitly selecting numeric columns
    # reviews: useful, funny, cool (no suffix as they don't collide with business)
    # business: stars_business, review_count, latitude, longitude, is_open
    # users: average_stars, review_count_user, useful_user, funny_user, cool_user, fans
    numeric_cols = [
        'useful', 'funny', 'cool',
        'stars_business', 'review_count', 'latitude', 'longitude', 'is_open',
        'average_stars', 'review_count_user', 'useful_user', 'funny_user', 'cool_user', 'fans'
    ]
    
    print(f"Using numeric features: {numeric_cols}")
    
    # Fill numeric NAs
    df_num = df[numeric_cols].fillna(0)
    
    # Scale
    if is_training:
        print("Fitting StandardScaler...")
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_num)
    else:
        print("Transforming numeric features...")
        X_num = scaler.transform(df_num)
        
    # Combine
    X = scipy.sparse.hstack((X_text, X_num))
    
    return X, y, vectorizer, scaler

def train_model():
    df = load_data()
    
    # Split Data manually to ensure we have a test set
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare Train Features
    X_train, y_train, vectorizer, scaler = prepare_features(train_df, is_training=True)
    
    # Prepare Test Features for validation during training (optional but good for checking)
    X_test, y_test, _, _ = prepare_features(test_df, vectorizer=vectorizer, scaler=scaler, is_training=False)
    
    # Train Linear Regression
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on test split...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # Round for "classification" metric
    y_pred_rounded = np.round(y_pred)
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    accuracy = accuracy_score(y_test, y_pred_rounded)
    
    print(f"MSE: {mse:.4f}")
    print(f"Accuracy (1-5 stars): {accuracy:.4f}")
    
    # Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Saving artifacts to {MODEL_DIR}...")
    with open(os.path.join(MODEL_DIR, 'linear_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Done.")

if __name__ == "__main__":
    train_model()
