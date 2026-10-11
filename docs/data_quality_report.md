# Data Quality and Preprocessing Audit Report

## 1. Cohort Overview
- Total ICU Observations: 7945
- Unique Sepsis Patients: 600
- Average Steps per Patient: 13.2 (4-hour bins, up to 72h observation window)
- Crude Mortality Rate: 3.33%

## 2. Feature Distribution and Missingness Summary

| Feature | Missing Count | Missing % | Min | Median | Max | Outliers (Tukey) |
| --- | --- | --- | --- | --- | --- | --- |
| hadm_id | 0 | 0.0% | 150000.00 | 150300.00 | 150599.00 | 0 |
| age | 0 | 0.0% | 25.00 | 64.00 | 90.00 | 60 |
| gender | 0 | 0.0% | 0.00 | 1.00 | 1.00 | 0 |
| elixhauser | 0 | 0.0% | 0.00 | 3.00 | 10.00 | 74 |
| sofa_initial | 0 | 0.0% | 2.00 | 6.00 | 16.00 | 11 |
| gcs | 0 | 0.0% | 8.00 | 15.00 | 15.00 | 479 |
| hr | 0 | 0.0% | 45.00 | 87.60 | 139.31 | 111 |
| sys_bp | 0 | 0.0% | 55.00 | 109.08 | 171.51 | 54 |
| dia_bp | 0 | 0.0% | 30.00 | 62.63 | 110.00 | 53 |
| map | 0 | 0.0% | 45.28 | 81.47 | 125.68 | 54 |
| rr | 0 | 0.0% | 8.00 | 20.01 | 42.00 | 28 |
| temp | 0 | 0.0% | 35.00 | 37.41 | 40.50 | 42 |
| spo2 | 0 | 0.0% | 77.17 | 95.18 | 100.00 | 66 |
| fio2 | 0 | 0.0% | 0.21 | 0.40 | 0.80 | 0 |
| lactate | 0 | 0.0% | 0.50 | 2.19 | 14.45 | 294 |
| creatinine | 0 | 0.0% | 0.30 | 1.68 | 9.18 | 408 |
| platelets | 0 | 0.0% | 15.00 | 219.84 | 415.19 | 51 |
| bilirubin | 0 | 0.0% | 0.10 | 1.18 | 7.73 | 343 |
| wbc | 0 | 0.0% | 1.00 | 12.07 | 27.71 | 9 |
| bun | 0 | 0.0% | 5.00 | 24.71 | 61.61 | 48 |
| glucose | 0 | 0.0% | 40.00 | 136.35 | 284.37 | 51 |
| potassium | 0 | 0.0% | 2.20 | 4.20 | 6.36 | 87 |
| sodium | 0 | 0.0% | 123.45 | 138.15 | 153.56 | 69 |
| chloride | 0 | 0.0% | 82.57 | 101.76 | 118.67 | 75 |
| bicarbonate | 0 | 0.0% | 10.00 | 21.90 | 36.16 | 14 |
| ph | 0 | 0.0% | 7.02 | 7.34 | 7.60 | 119 |
| pao2 | 0 | 0.0% | 45.00 | 96.05 | 176.97 | 45 |
| paco2 | 0 | 0.0% | 18.00 | 39.58 | 64.00 | 21 |
| hemoglobin | 0 | 0.0% | 4.59 | 10.48 | 18.00 | 62 |
| urine_output | 0 | 0.0% | 0.00 | 181.32 | 476.54 | 35 |
| cumulated_balance | 0 | 0.0% | -3585.35 | 633.89 | 5538.72 | 353 |
| iv_fluids | 0 | 0.0% | 0.00 | 138.47 | 2000.00 | 293 |
| vasopressors | 0 | 0.0% | 0.00 | 0.00 | 1.49 | 1018 |

## 3. Data Leakage and Integrity Verification
- Temporal ordering: Verified monotonically increasing step index per trajectory.
- Subject independence: Train/Val/Test splitting strictly enforced at subject_id level.
- Imputation & Scaling: Standard deviations and medians fitted exclusively on training cohort.
- State encoding: Centroids fitted on training observations only.