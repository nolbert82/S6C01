# Yelp Review Analysis: ML, Deep Learning, and Agentic AI

This project focuses on the intelligent analysis of the Yelp Open Dataset. The goal is to predict review ratings (1-5 stars), determine sentiment polarity (positive, negative, or neutral), and perform Aspect-Based Sentiment Analysis (ABSA) using Large Language Models (LLMs) and agentic frameworks.

## Project Structure

- `data/`: Contains raw, converted (CSV), and prepared datasets.
- `scripts/`: Utility scripts for data conversion and preparation.
- `text_representations/`: Scripts to train text vectorizers (TF-IDF, CountVectorizer).
- `models/`: Implementations of various ML and Deep Learning models (SVM, MLP, CNN, BERT, etc.).
- `llm/`: LLM-based approaches for zero-shot and few-shot classification.

## Installation

This project uses `uv` for Python package management. To install dependencies:

```bash
uv pip install -r requirements.txt
```

## Data Setup

To get started, you need the Yelp Open Dataset JSON files.

1. Create the `data/raw/` directory if it doesn't exist.
2. Place the following files in `data/raw/`:
   - `yelp_academic_dataset_review.json`
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_user.json`

## Execution Workflow

Follow these steps in order to process the data and train the models:

### 1. Data Conversion
Convert the raw JSON files into CSV format:
```bash
python scripts/convert_data_to_csv/convert_reviews4students.py
python scripts/convert_data_to_csv/convert_business.py
python scripts/convert_data_to_csv/convert_users.py
```
This will generate CSV files in `data/csv/`.

### 2. Data Preparation
Merge the datasets and prepare the final training and testing sets:
```bash
python scripts/prepare_data/prepare_training_data.py
```
This generates `training_dataset.csv` and `testing_dataset.csv` in `data/prepared/`.

### 3. Text Representation Training
Fit the vectorizers that will be shared across different models:
```bash
python text_representations/TFIDF/train.py
python text_representations/CountVectorizer/train.py
```

### 4. Model Training
Navigate to the desired model directory and run the training script. For example, to train the TF-IDF + MLP model for score prediction:
```bash
python models/tfidf_mlp/score_prediction/train.py
```
Trained models and results will be saved in the `saved_model/` subdirectory of the respective model.

### 5. Model Testing
Run the testing script to evaluate the model on the test set:
```bash
python models/tfidf_mlp/score_prediction/test.py
```

### 6. LLM Inferences
For zero-shot or few-shot predictions using LLMs, look into the `llm/` directory:
```bash
python llm/score_prediction/zero_shot/test.py
```

## Tasks Covered
- **Polarity Prediction**: Classifying reviews as Positive, Negative, or Neutral.
- **Score Prediction**: Predicting the exact star rating (1 to 5).
- **Aspect Extraction**: Identifying specific features (food, service, price) and their associated sentiment.
