import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

def label_sentiment(stars):
    # Map 1-5 stars to -2 to 2 scale
    # 1: -2 (Very Negative)
    # 2: -1 (Negative)
    # 3:  0 (Neutral)
    # 4:  1 (Positive)
    # 5:  2 (Very Positive)
    return float(stars - 3)

class YelpDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # For regression, labels must be floats
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.flatten() # For regression, shape is (batch, 1) or (batch,)
    
    # MSE for regression quality
    mse = ((preds - labels) ** 2).mean()
    
    # Accuracy based on polarity (Sign of the values)
    # Neutral (0) is treated separately or as its own category if 0
    # Polarity accuracy: same sign
    # We can discretize both to -1, 0, 1 for polarity accuracy
    def get_polarity(val):
        if val < -0.5: return -1
        if val > 0.5: return 1
        return 0
    
    # vectorized version
    polar_labels = [1 if l > 0.5 else (-1 if l < -0.5 else 0) for l in labels]
    polar_preds = [1 if p > 0.5 else (-1 if p < -0.5 else 0) for p in preds]
    
    acc = accuracy_score(polar_labels, polar_preds)
    
    return {
        'mse': mse.item(),
        'accuracy': acc,
    }

def train():
    # 1. Charger les données
    data_path = "./data/csv/yelp_academic_reviews4students.csv"
    print(f"Chargement des données depuis {data_path}...")
    
    # Utilisation d'un sous-ensemble pour la démonstration (ajuster nrows au besoin)
    df = pd.read_csv(data_path, nrows=5000)
    
    # Prétraitement simple
    df = df.dropna(subset=['text', 'stars'])
    df['label'] = df['stars'].apply(label_sentiment)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )

    # 2. Préparer le tokenizer et le modèle
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = YelpDataset(train_encodings, train_labels)
    val_dataset = YelpDataset(val_encodings, val_labels)

    # For regression, num_labels=1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Détection et configuration du device (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    if device.type == "cuda":
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    
    model.to(device)

    # 3. Paramètres d'entraînement
    output_dir = "./polarity_model/model/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
    )

    # 4. Entraînement
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Début de l'entraînement...")
    trainer.train()

    # 5. Sauvegarder le modèle final
    final_model_dir = "./polarity_model/model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Modèle et tokenizer sauvegardés dans {final_model_dir}")

if __name__ == "__main__":
    train()
