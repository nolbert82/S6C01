import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def artifacts_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def main():
    output_dir = artifacts_dir()

    with open(output_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(output_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    test_df_path = output_dir / "test_split.csv"
    if not test_df_path.exists():
        raise FileNotFoundError(
            "Missing test split file. Run train.py first to generate artifacts/test_split.csv."
        )

    import pandas as pd

    test_df = pd.read_csv(test_df_path)

    x_test = vectorizer.transform(test_df["combined_text"])
    y_test = test_df["label"]
    y_pred = model.predict(x_test)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
