import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

FEATURE_COLS = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

def main():
    b3db = pd.read_csv("data/processed/b3db_features.csv")
    bbbp = pd.read_csv("data/processed/bbbp_features.csv")

    X = b3db[FEATURE_COLS]
    y = b3db["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    xgb_grid = {
        "max_depth": [3, 4, 5],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "reg_lambda": [1, 5, 10],  # L2 regularization — higher = less overfitting
    }
    grid_search = GridSearchCV(
        XGBClassifier(eval_metric="logloss"), xgb_grid, scoring="roc_auc", cv=5
    )
    grid_search.fit(X_train, y_train)
    print("Best params:", grid_search.best_params_)
    print("Best internal ROC-AUC:", grid_search.best_score_)

    X_bbbp = bbbp[FEATURE_COLS]
    y_bbbp = bbbp["label"]
    tuned_bbbp_auc = roc_auc_score(y_bbbp, grid_search.best_estimator_.predict_proba(X_bbbp)[:, 1])
    print("Tuned XGB BBBP ROC-AUC:", tuned_bbbp_auc)

    # Only overwrite the saved model if this genuinely beats what train_model.py produced
    joblib.dump(grid_search.best_estimator_, "models/bbb_classifier_tuned_xgb.pkl")
    print("Saved to models/bbb_classifier_tuned_xgb.pkl — compare its BBBP score to models/bbb_classifier.pkl before switching")

if __name__ == "__main__":
    main()