import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import Preprocessor

def train_and_explain_with_shap(
    df: pd.DataFrame,
    target_col: str = "product_wg_ton",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    1) Drop rows with missing target AND known bad rows
    2) Split into train/test
    3) Preprocess with Preprocessor
    4) Fit XGB, evaluate, and produce SHAP summary plot
    """
    # 1a) Drop missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # 1b) Drop the “bad” rows: storage_issue_reported_l3m == 0 AND no certificate
    bad_mask = (
        (df['storage_issue_reported_l3m'] == 0) &
        (df['approved_wh_govt_certificate'].isna())
    )
    num_bad = bad_mask.sum()
    print(f"Dropping {num_bad} bad rows before split")
    df = df.loc[~bad_mask].reset_index(drop=True)

    # 2) Split
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3) Preprocess
    prep = Preprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p  = prep.transform(X_test)

    # 4) Fit XGBoost
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="rmse",
        verbosity=0
    )
    model.fit(X_train_p, y_train)

    # 5) Evaluate
    preds = model.predict(X_test_p)
    print("XGBoost performance:")
    print(f"  MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"  MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"  R² : {r2_score(y_test, preds):.4f}")

    # 6) SHAP setup
    num_cols = prep.num_cols
    ohe = prep.preprocessor.named_transformers_['cat'] \
               .named_steps['ohe']
    cat_cols = ohe.get_feature_names_out(prep.cat_cols).tolist()
    feature_names = num_cols + cat_cols

    # 7) Compute SHAP values & plot
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    plt.figure(figsize=(12,6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    return model, prep, explainer