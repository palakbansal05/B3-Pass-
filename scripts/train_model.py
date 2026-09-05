import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

FEATURE_COLS = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

def main():
    b3db = pd.read_csv("data/processed/b3db_features.csv")
    bbbp = pd.read_csv("data/processed/bbbp_features.csv")

    X = b3db[FEATURE_COLS]
    y = b3db["label"]
    # --- 5-fold CV diagnostic: is the 0.948 single-split score stable, or a lucky split? ---
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=42), X, y, cv=5, scoring="roc_auc"
    )
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    # --- end diagnostic block ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgb": XGBClassifier(eval_metric="logloss"),
    }

    best_name, best_auc, best_model = None, 0, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        print(f"{name} ROC-AUC: {auc:.4f}")
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, model

    print(f"\nBest model: {best_name} (ROC-AUC {best_auc:.4f})")
    joblib.dump(best_model, "models/bbb_classifier.pkl")

    # Validate on cleaned, deduped BBBP — the real generalization test
    X_bbbp = bbbp[FEATURE_COLS]
    y_bbbp = bbbp["label"]
    bbbp_preds = best_model.predict_proba(X_bbbp)[:, 1]
    print("BBBP validation ROC-AUC:", roc_auc_score(y_bbbp, bbbp_preds))

if __name__ == "__main__":
    main()