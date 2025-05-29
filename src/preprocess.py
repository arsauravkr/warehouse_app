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
      2) Engineers wh_age = 2024 - wh_est_year (if present), then drops wh_est_year
      3) Median-imputes all numeric columns
      4) Standard-scales numeric features
      5) Imputes categoricals as 'Missing' and one-hot encodes them
    """
    def __init__(self):
        self.id_cols      = ['Ware_house_ID', 'WH_Manager_ID']
        self.year_col     = 'wh_est_year'
        self.current_year = 2024

        # to be set in fit()
        self.num_cols     = None
        self.cat_cols     = None
        self.preprocessor = None

    def fit(self, X, y=None):
        df = self._clean_and_engineer(X)

        # Identify numeric vs categorical
        self.num_cols = df.select_dtypes(include='number').columns.tolist()
        if 'product_wg_ton' in self.num_cols:
            self.num_cols.remove('product_wg_ton')
        self.cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline,     self.num_cols),
            ('cat', categorical_pipeline, self.cat_cols),
        ], remainder='drop')

        # Fit transformers
        self.preprocessor.fit(df)
        return self

    def transform(self, X):
        df = self._clean_and_engineer(X)
        return self.preprocessor.transform(df)

    def _clean_and_engineer(self, X):
        df = X.copy()

        # 1) Drop duplicates & ID columns
        df = df.drop_duplicates()
        df.drop(columns=[c for c in self.id_cols if c in df.columns],
                inplace=True, errors='ignore')

        # 2) Engineer wh_age if wh_est_year exists
        if self.year_col in df.columns:
            df['wh_age'] = self.current_year - df[self.year_col]
            df.drop(columns=[self.year_col], inplace=True, errors='ignore')

        # 3) Impute any remaining numeric NaNs
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].median())

        return df