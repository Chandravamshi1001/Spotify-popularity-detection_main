# src/train.py
import os
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocess import load_data, clean_and_select, make_target, split_and_scale

def train(args):
    # 1️⃣ Load data
    df = load_data(args.data_path)
    df = clean_and_select(df)

    # 2️⃣ Create target column
    df = make_target(df, threshold=args.threshold)

    # 3️⃣ Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, test_size=args.test_size)

    # ✅ Debug: check shapes
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 4️⃣ Train RandomForest
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5️⃣ Predict and evaluate
    preds = model.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

    # 6️⃣ Save model and scaler
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "rf_model.joblib")
    scaler_path = os.path.join(args.model_dir, "scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tracks.csv", help="Path to CSV dataset")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model and scaler")
    parser.add_argument("--threshold", type=int, default=60, help="Popularity threshold to classify popular songs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--n_estimators", type=int, default=200, help="RandomForest n_estimators")
    parser.add_argument("--max_depth", type=int, default=None, help="RandomForest max depth")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(args)
