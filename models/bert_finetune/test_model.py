import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

def label_sentiment(stars):
    # Map 1-5 stars to -2 to 2 scale
    return float(stars - 3)

def test():
    model_path = "./polarity_model/model"
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle n'a pas été trouvé dans {model_path}. Veuillez d'abord exécuter train_model.py.")
        return

    print("Chargement du modèle et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Créer un pipeline pour une utilisation facile
    # For regression models, pipeline returns a single score
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # 1. Test sur des exemples manuels
    print("\n--- Tests sur des exemples manuels ---")
    examples = [
        "This restaurant was amazing! The food was delicious and service was fast.",
        "It was okay, nothing special. The wait was a bit long.",
        "Terrible experience. The staff was rude and the place was dirty."
    ]
    
    def get_friendly_label(score):
        if score < -0.5: return "Négatif"
        if score > 0.5: return "Positif"
        return "Neutre"
    
    for text in examples:
        result = sentiment_pipeline(text)[0]
        # Regression models in pipeline usually return the value in 'score' if num_labels=1
        # and the label is "LABEL_0"
        score = result['score'] if result['label'] == 'LABEL_0' else 0.0
        # Wait, actually for regression num_labels=1, pipeline might behave differently
        # Let's check how it handles it. Usually it's result['score']
        
        friendly_label = get_friendly_label(score)
        print(f"Texte: {text}")
        print(f"Prédiction brute: {score:.4f} -> {friendly_label}\n")

    # 2. Évaluation sur un échantillon de données test
    data_path = "./data/csv/yelp_academic_reviews4students.csv"
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
        
        # Pour la régression, on convertit les scores en classes pour le rapport
        def score_to_class(score):
            if score < -0.5: return 0 # Négatif
            if score > 0.5: return 2 # Positif
            return 1 # Neutre

        preds_scores = [res['score'] for res in results]
        predictions = [score_to_class(s) for s in preds_scores]
        actual_classes = [score_to_class(a) for a in actuals]
        
        print("\n--- Rapport de Classification ---")
        print(classification_report(actual_classes, predictions, target_names=["Négatif", "Neutre", "Positif"], labels=[0, 1, 2]))
        print(f"Précision globale (basée sur polarité): {accuracy_score(actual_classes, predictions):.4f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {e}")

if __name__ == "__main__":
    test()
