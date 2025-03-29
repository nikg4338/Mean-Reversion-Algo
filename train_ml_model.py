import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    input_file = "updated_historical_stock_data.xlsx"
    output_file = "ml_training_results.xlsx"

    print(f"\n🔍 Loading data from '{input_file}' for model training...")

    try:
        df = pd.read_excel(input_file, sheet_name="Market Screener Predictions")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    required_cols = {"symbol", "date", "close", "RSI", "Z_Score", "Sharpe_Ratio", "Pairwise_Z", "Trade Action"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # ✅ Correct label mapping (BUY=0, HOLD=1, SELL=2)
    label_map = {"BUY": 0, "HOLD": 1, "SELL": 2}
    df["Target"] = df["Trade Action"].map(label_map)

    # Drop any rows where mapping failed
    df = df.dropna(subset=["Target"])
    df["Target"] = df["Target"].astype(int)

    # 📊 Print target class distribution
    class_counts = df["Target"].value_counts().sort_index()
    print("\n📊 Target Class Distribution:")
    print(class_counts.rename(index={0: "BUY", 1: "HOLD", 2: "SELL"}))

    # ⚠️ Sanity check: must have at least 2 different classes to train
    if class_counts.shape[0] < 2:
        print("\n⚠️ Not enough class variety to train. Need at least 2 classes.")
        return

    # Prepare training data
    features = ["RSI", "Z_Score", "Sharpe_Ratio", "Pairwise_Z"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 🧠 Train model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=["BUY", "HOLD", "SELL"]))

    # 💾 Save model
    try:
        joblib.dump(model, "ml_model_trained.pkl")
        print("✅ Model saved as 'ml_model_trained.pkl'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    # 💡 Save full predictions for inspection
    probs = model.predict_proba(X)
    df["Predicted"] = model.predict(X)
    df["Confidence Score"] = probs.max(axis=1)
    df["Predicted Action"] = df["Predicted"].map({0: "BUY", 1: "HOLD", 2: "SELL"})

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="ML Predictions", index=False)
        print(f"✅ Results saved to '{output_file}' [ML Predictions]")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()
