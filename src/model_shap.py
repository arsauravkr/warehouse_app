import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import Preprocessor

def train_and_explain_with_shap(
    csv_path="raw_data.csv",
    target_col="product_wg_ton",
    test_size=0.2,
    random_state=42
):
    # 1) Load and split
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Preprocess
    prep = Preprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p  = prep.transform(X_test)

    # 3) Fit XGBoost
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

    # 4) Evaluate
    preds = model.predict(X_test_p)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print("XGBoost performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R² : {r2:.4f}")

    # 5) SHAP explanation
    # Build feature name list: numeric + one-hot feature names
    num_cols = prep.num_cols
    ohe = prep.preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_cols = ohe.get_feature_names_out(prep.cat_cols)
    feature_names = num_cols + list(cat_cols)

    # Convert test set to DataFrame for SHAP
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)

    # Create SHAP explainer and values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # 6) Summary plot
    plt.figure(figsize=(12,6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.show()

    return model, prep, explainer

if __name__ == "__main__":
    train_and_explain_with_shap()