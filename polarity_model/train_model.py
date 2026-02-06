import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

def label_sentiment(stars):
    if stars <= 2:
        return 0  # Négatif
    elif stars == 3:
        return 1  # Neutre
    else:
        return 2  # Positif

class YelpDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
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

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

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
