# Data Quality and Preprocessing Audit Report

## 1. Cohort Overview
- Total ICU Observations: 7840
- Unique patients: 600
- Source: synthetic coursework demo. This audit does not verify Sepsis-3 eligibility.
- Average Steps per Patient: 13.1
- Crude Mortality Rate: 4.83%

## 2. Feature Distribution and Missingness Summary

| Feature | Missing Count | Missing % | Min | Median | Max | Outliers (Tukey) |
| --- | --- | --- | --- | --- | --- | --- |
| age | 0 | 0.0% | 27.00 | 64.00 | 90.00 | 97 |
| gender | 0 | 0.0% | 0.00 | 1.00 | 1.00 | 0 |
| elixhauser | 0 | 0.0% | 0.00 | 3.00 | 10.00 | 97 |
| sofa_initial | 0 | 0.0% | 2.00 | 6.00 | 16.00 | 12 |
| gcs | 0 | 0.0% | 8.00 | 15.00 | 15.00 | 1010 |
| hr | 0 | 0.0% | 45.00 | 87.17 | 135.23 | 36 |
| sys_bp | 0 | 0.0% | 55.00 | 128.31 | 190.00 | 6 |
| dia_bp | 0 | 0.0% | 30.00 | 80.03 | 110.00 | 0 |
| map | 0 | 0.0% | 52.67 | 96.10 | 136.67 | 0 |
| rr | 0 | 0.0% | 8.00 | 19.98 | 42.00 | 66 |
| temp | 0 | 0.0% | 35.00 | 37.52 | 40.50 | 27 |
| spo2 | 0 | 0.0% | 77.83 | 95.83 | 100.00 | 114 |
| fio2 | 0 | 0.0% | 0.21 | 0.40 | 0.80 | 0 |
| lactate | 0 | 0.0% | 0.50 | 2.25 | 12.36 | 325 |
| creatinine | 0 | 0.0% | 0.30 | 1.53 | 9.44 | 355 |
| platelets | 0 | 0.0% | 15.00 | 223.51 | 414.18 | 79 |
| bilirubin | 0 | 0.0% | 0.10 | 1.18 | 6.12 | 363 |
| wbc | 0 | 0.0% | 1.00 | 12.51 | 32.01 | 34 |
| bun | 0 | 0.0% | 5.00 | 24.59 | 54.54 | 1 |
| glucose | 0 | 0.0% | 40.00 | 135.72 | 257.28 | 20 |
| potassium | 0 | 0.0% | 2.41 | 4.16 | 5.80 | 72 |
| sodium | 0 | 0.0% | 123.88 | 138.05 | 154.97 | 80 |
| chloride | 0 | 0.0% | 84.04 | 102.06 | 120.86 | 20 |
| bicarbonate | 0 | 0.0% | 10.00 | 21.67 | 35.96 | 91 |
| ph | 0 | 0.0% | 7.08 | 7.34 | 7.60 | 29 |
| pao2 | 0 | 0.0% | 45.00 | 95.57 | 161.74 | 24 |
| paco2 | 0 | 0.0% | 18.13 | 40.60 | 66.94 | 38 |
| hemoglobin | 0 | 0.0% | 4.50 | 10.39 | 18.00 | 102 |
| urine_output | 0 | 0.0% | 0.00 | 180.85 | 470.97 | 58 |
| cumulated_balance | 0 | 0.0% | -3305.88 | 549.89 | 5135.77 | 361 |
| iv_fluids | 0 | 0.0% | 0.00 | 120.07 | 2000.00 | 349 |
| vasopressors | 0 | 0.0% | 0.00 | 0.00 | 1.29 | 1238 |

## 3. Data Leakage and Integrity Verification
- Temporal ordering and contiguous steps: PASS
- Consistent binary outcome per episode: PASS
- Patient-disjoint splits covering raw cohort: PASS
- Model fit provenance is not verifiable from raw CSV; inspect pipeline fit calls and regression tests.