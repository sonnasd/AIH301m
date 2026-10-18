import os
import numpy as np
import pandas as pd

def run_data_quality_audit(input_csv="data/raw_sepsis_cohort.csv", report_md="docs/data_quality_report.md", splits=None, data_source="unspecified"):
    os.makedirs(os.path.dirname(os.path.abspath(report_md)), exist_ok=True)
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} does not exist")
        
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Cannot audit an empty cohort")
    episode_cols = ['subject_id'] + [c for c in ['hadm_id', 'icustay_id'] if c in df]
    temporal_ok = not df[episode_cols].isna().any().any()
    outcomes_ok = df['died'].isin([0, 1]).all()
    for _, group in df.groupby(episode_cols, sort=False):
        steps = group['step'].to_numpy(dtype=float)
        temporal_ok &= bool(np.isfinite(steps).all() and (steps >= 0).all() and (steps % 1 == 0).all() and (np.diff(steps) == 1).all())
        outcomes_ok &= group['died'].nunique(dropna=False) == 1
    split_status = 'Not checked: split data not supplied'
    if splits is not None:
        ids = [set(part['subject_id']) for part in splits]
        split_ok = (
            len(ids) == 3 and all(ids) and
            all(not left.intersection(right) for i, left in enumerate(ids) for right in ids[i + 1:]) and
            set.union(*ids) == set(df['subject_id'])
        )
        split_status = 'PASS' if split_ok else 'FAIL'
    
    total_records = len(df)
    n_patients = df['subject_id'].nunique()
    
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c not in ['subject_id', 'hadm_id', 'icustay_id', 'step', 'died']]
    
    missing_summary = []
    for c in feature_cols:
        n_missing = int(df[c].isna().sum())
        pct_missing = float(n_missing / total_records * 100)
        c_min = float(df[c].min()) if n_missing < total_records else np.nan
        c_median = float(df[c].median()) if n_missing < total_records else np.nan
        c_max = float(df[c].max()) if n_missing < total_records else np.nan
        
        q25 = df[c].quantile(0.25)
        q75 = df[c].quantile(0.75)
        iqr = q75 - q25
        n_outliers = int(((df[c] < q25 - 1.5 * iqr) | (df[c] > q75 + 1.5 * iqr)).sum())
        
        missing_summary.append({
            'Feature': c,
            'Missing Count': n_missing,
            'Missing %': f"{pct_missing:.1f}%",
            'Min': f"{c_min:.2f}",
            'Median': f"{c_median:.2f}",
            'Max': f"{c_max:.2f}",
            'Outliers (Tukey)': n_outliers
        })
        
    summary_df = pd.DataFrame(missing_summary)
    
    headers = list(summary_df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in summary_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    table_str = "\n".join(table_lines)
    
    report_lines = [
        "# Data Quality and Preprocessing Audit Report",
        "",
        "## 1. Cohort Overview",
        f"- Total ICU Observations: {total_records}",
        f"- Unique patients: {n_patients}",
        f"- Source: {data_source}. This audit does not verify Sepsis-3 eligibility.",
        f"- Average Steps per Patient: {total_records / n_patients:.1f}",
        f"- Crude Mortality Rate: {df.groupby('subject_id')['died'].first().mean() * 100:.2f}%",
        "",
        "## 2. Feature Distribution and Missingness Summary",
        "",
        table_str,
        "",
        "## 3. Data Leakage and Integrity Verification",
        f"- Temporal ordering and contiguous steps: {'PASS' if temporal_ok else 'FAIL'}",
        f"- Consistent binary outcome per episode: {'PASS' if outcomes_ok else 'FAIL'}",
        f"- Patient-disjoint splits covering raw cohort: {split_status}",
        "- Model fit provenance is not verifiable from raw CSV; inspect pipeline fit calls and regression tests."
    ]
    
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Data quality audit written to {report_md}")
    return summary_df

if __name__ == '__main__':
    run_data_quality_audit()
