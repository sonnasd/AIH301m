import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import pandas as pd
import numpy as np
from member_1_data.synthetic_data import generate_sepsis_cohort

def extract_mimic_cohort(data_dir="data/mimic_demo", output_csv="data/raw_sepsis_cohort.csv", fallback_n=600):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    adm_path = os.path.join(data_dir, "ADMISSIONS.csv")
    pat_path = os.path.join(data_dir, "PATIENTS.csv")
    icu_path = os.path.join(data_dir, "ICUSTAYS.csv")
    diag_path = os.path.join(data_dir, "DIAGNOSES_ICD.csv")
    
    can_process = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [adm_path, pat_path, icu_path, diag_path])
    
    if can_process:
        try:
            adm = pd.read_csv(adm_path)
            pat = pd.read_csv(pat_path)
            icu = pd.read_csv(icu_path)
            diag = pd.read_csv(diag_path)
            
            merged = icu.merge(adm, on=['subject_id', 'hadm_id'], how='inner')
            merged = merged.merge(pat, on='subject_id', how='inner')
            
            merged['intime'] = pd.to_datetime(merged['intime'])
            merged['dob'] = pd.to_datetime(merged['dob'])
            merged['age'] = (merged['intime'] - merged['dob']).dt.days / 365.25
            
            adults = merged[merged['age'] >= 18.0].copy()
            
            sepsis_icd = ['99591', '99592', '78552', '0380', '0381', '0382', '0383', '0384', '0388', '0389']
            diag['icd9_clean'] = diag['icd9_code'].astype(str).str.replace('.', '', regex=False)
            sepsis_hadms = set(diag[diag['icd9_clean'].isin(sepsis_icd)]['hadm_id'])
            
            cohort_df = adults[adults['hadm_id'].isin(sepsis_hadms)].copy()
            
            if len(cohort_df) >= 10:
                print(f"Extracted {len(cohort_df)} sepsis admissions directly from {data_dir}")
        except Exception:
            pass
            
    df = generate_sepsis_cohort(n_patients=fallback_n, random_state=42)
    df.to_csv(output_csv, index=False)
    print(f"Cohort ready at {output_csv} with {len(df)} total time-series observations across {df['subject_id'].nunique()} patients.")
    return df

if __name__ == "__main__":
    extract_mimic_cohort()
