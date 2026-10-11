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
        rng = np.random.RandomState(random_state)
        unique_subjects = df['subject_id'].unique()
        rng.shuffle(unique_subjects)
        
        n_total = len(unique_subjects)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
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
        active_cols = [c for c in self.feature_cols if c in train_df.columns]
        self.feature_cols = active_cols
        
        for col in self.feature_cols:
            self.medians[col] = float(train_df[col].median())
            
        filled = train_df[self.feature_cols].fillna(self.medians)
        self.scaler.fit(filled)
        self.is_fitted = True
        return self
        
    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
            
        df_out = df.copy()
        filled = df_out[self.feature_cols].fillna(self.medians)
        scaled_features = self.scaler.transform(filled)
        
        for idx, col in enumerate(self.feature_cols):
            df_out[col + '_scaled'] = scaled_features[:, idx]
            
        return df_out
        
    def fit_transform(self, train_df):
        self.fit(train_df)
        return self.transform(train_df)
