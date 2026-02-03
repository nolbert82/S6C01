import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

def label_sentiment(stars):
    if stars <= 2:
        return 0  # Négatif
    elif stars == 3:
        return 1  # Neutre
    else:
        return 2  # Positif

def test():
    model_path = "c:/Travail/S6C01/polarity_model/model"
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle n'a pas été trouvé dans {model_path}. Veuillez d'abord exécuter train_model.py.")
        return

    print("Chargement du modèle et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Créer un pipeline pour une utilisation facile
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # 1. Test sur des exemples manuels
    print("\n--- Tests sur des exemples manuels ---")
    examples = [
        "This restaurant was amazing! The food was delicious and service was fast.",
        "It was okay, nothing special. The wait was a bit long.",
        "Terrible experience. The staff was rude and the place was dirty."
    ]
    
    # Mapping des labels pour l'affichage
    label_map = {"LABEL_0": "Négatif", "LABEL_1": "Neutre", "LABEL_2": "Positif"}
    
    for text in examples:
        result = sentiment_pipeline(text)[0]
        friendly_label = label_map.get(result['label'], result['label'])
        print(f"Texte: {text}")
        print(f"Prédiction: {friendly_label} (Score: {result['score']:.4f})\n")

    # 2. Évaluation sur un échantillon de données test
    data_path = "c:/Travail/S6C01/data/csv/yelp_academic_reviews4students.csv"
    print(f"Évaluation sur un échantillon de données de {data_path}...")
    
    # On saute les 5000 premières lignes utilisées pour l'entraînement
    try:
        df_test = pd.read_csv(data_path, skiprows=range(1, 5001), nrows=500)
        # Remettre les headers car skiprows les supprime si on ne fait pas attention
        header_df = pd.read_csv(data_path, nrows=0)
        df_test.columns = header_df.columns
        
        df_test = df_test.dropna(subset=['text', 'stars'])
        df_test['actual_label'] = df_test['stars'].apply(label_sentiment)
        
        texts = df_test['text'].tolist()
        actuals = df_test['actual_label'].tolist()
        
        print("Inférence sur 500 exemples de test...")
        results = sentiment_pipeline(texts, truncation=True, padding=True)
        
        # Extraire l'index du label (LABEL_0 -> 0)
        predictions = [int(res['label'].split('_')[1]) for res in results]
        
        print("\n--- Rapport de Classification ---")
        print(classification_report(actuals, predictions, target_names=["Négatif", "Neutre", "Positif"]))
        print(f"Précision globale: {accuracy_score(actuals, predictions):.4f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {e}")

if __name__ == "__main__":
    test()
