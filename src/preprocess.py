import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessing pipeline that:
      1) Drops duplicate rows and identifier columns
      2) Filters out rows with storage_issue_reported_l3m == 0 and missing certificates
      3) Engineers wh_age = 2024 - wh_est_year, then drops wh_est_year
      4) Median-imputes all numeric columns
      5) Standard-scales numeric features
      6) Imputes categoricals as 'Missing' and one-hot encodes them
    """
    def __init__(self):
        # Columns to drop during cleaning
        self.id_cols      = ['Ware_house_ID', 'WH_Manager_ID']
        # Columns used in filtering error rows
        self.issue_col    = 'storage_issue_reported_l3m'
        self.cert_col     = 'approved_wh_govt_certificate'
        # Year feature for age engineering
        self.year_col     = 'wh_est_year'
        self.current_year = 2024

        # placeholders to be set in fit()
        self.num_cols     = None
        self.cat_cols     = None
        self.preprocessor = None

    def fit(self, X, y=None):
        # 1–3) Clean and feature engineer
        df = self._clean_and_engineer(X)

        # 4) Identify numeric vs categorical
        self.num_cols = df.select_dtypes(include='number').columns.tolist()
        if 'product_wg_ton' in self.num_cols:
            self.num_cols.remove('product_wg_ton')
        self.cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

        # 5) Build sub-pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            # use sparse_output for newer sklearn
            ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])


        # 6) Combine
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline,     self.num_cols),
            ('cat', categorical_pipeline, self.cat_cols),
        ], remainder='drop')

        # 7) Fit on cleaned DataFrame
        self.preprocessor.fit(df)
        return self

    def transform(self, X):
        df = self._clean_and_engineer(X)
        return self.preprocessor.transform(df)

    def _clean_and_engineer(self, X):
        df = X.copy()

        # 1) Drop duplicates & ID cols
        df = df.drop_duplicates()
        df.drop(columns=[c for c in self.id_cols if c in df.columns],
                inplace=True, errors='ignore')

        # 2) Filter out the 908 bad rows
        mask_bad = (df[self.issue_col] == 0) & df[self.cert_col].isna()
        df = df.loc[~mask_bad].reset_index(drop=True)

        # 3) Engineer wh_age and drop original year
        df['wh_age'] = self.current_year - df[self.year_col]
        df.drop(columns=[self.year_col], inplace=True, errors='ignore')

        # 4) Median-impute any remaining numeric NaNs
        for col in df.select_dtypes(include='number').columns:
            df[col].fillna(df[col].median(), inplace=True)

        return df