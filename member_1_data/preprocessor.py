import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ClinicalPreprocessor:
    def __init__(self, feature_cols=None):
        if feature_cols is None:
            self.feature_cols = [
                'age', 'gender', 'elixhauser', 'sofa_initial', 'gcs',
                'hr', 'sys_bp', 'dia_bp', 'map', 'rr', 'temp', 'spo2', 'fio2',
                'lactate', 'creatinine', 'platelets', 'bilirubin', 'wbc',
                'bun', 'glucose', 'potassium', 'sodium', 'chloride', 'bicarbonate',
                'ph', 'pao2', 'paco2', 'hemoglobin', 'urine_output', 'cumulated_balance'
            ]
        else:
            self.feature_cols = feature_cols
            
        self.scaler = StandardScaler()
        self.medians = {}
        self.is_fitted = False
        
    def split_patients(self, df, train_ratio=0.70, val_ratio=0.15, random_state=42):
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 - train_ratio:
            raise ValueError("Train, validation and test ratios must be positive")
        if df.empty or df['subject_id'].isna().any():
            raise ValueError("Patient identifiers must be present")
        rng = np.random.RandomState(random_state)
        unique_subjects = df['subject_id'].unique()
        rng.shuffle(unique_subjects)
        
        n_total = len(unique_subjects)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        if min(n_train, n_val, n_total - n_train - n_val) < 1:
            raise ValueError("Not enough patients for three nonempty splits")
        
        train_ids = set(unique_subjects[:n_train])
        val_ids = set(unique_subjects[n_train:n_train + n_val])
        test_ids = set(unique_subjects[n_train + n_val:])
        
        train_df = df[df['subject_id'].isin(train_ids)].copy().reset_index(drop=True)
        val_df = df[df['subject_id'].isin(val_ids)].copy().reset_index(drop=True)
        test_df = df[df['subject_id'].isin(test_ids)].copy().reset_index(drop=True)
        
        train_df['dataset_split'] = 'train'
        val_df['dataset_split'] = 'val'
        test_df['dataset_split'] = 'test'
        
        return train_df, val_df, test_df
        
    def fit(self, train_df):
        self._validate_features(train_df)
        for col in self.feature_cols:
            self.medians[col] = float(train_df[col].median())
        if not np.isfinite(list(self.medians.values())).all():
            raise ValueError("Training features cannot be entirely missing")
            
        filled = train_df[self.feature_cols].fillna(self.medians)
        self.scaler.fit(filled)
        self.is_fitted = True
        return self

    def _validate_features(self, df):
        missing = set(self.feature_cols) - set(df.columns)
        if missing or df.empty:
            raise ValueError(f"Empty data or missing features: {sorted(missing)}")
        if np.isinf(df[self.feature_cols].to_numpy(dtype=float)).any():
            raise ValueError("Infinite feature values are not allowed")
        
    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        self._validate_features(df)
            
        df_out = df.copy()
        filled = df_out[self.feature_cols].fillna(self.medians)
        scaled_features = self.scaler.transform(filled)
        
        for idx, col in enumerate(self.feature_cols):
            df_out[col + '_scaled'] = scaled_features[:, idx]
            
        return df_out
        
    def fit_transform(self, train_df):
        self.fit(train_df)
        return self.transform(train_df)
