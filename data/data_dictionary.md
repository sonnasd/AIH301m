# Clinical Data Dictionary - Sepsis-3 MDP Formulation

This data dictionary defines all clinical variables, transformations, action discretizations, and MDP representations used by the 3 project modules.

---

## 1. Demographics and Baseline Scores

| Variable Name | Original Source | Units | Aggregation / Handling | Clinical Meaning |
|---|---|---|---|---|
| `subject_id` | PATIENTS.csv | Integer | ID | Unique patient identifier (de-identified) |
| `icustay_id` | ICUSTAYS.csv | Integer | ID | Unique ICU admission episode |
| `step` | Time series index | Integer | 0..17 | 4-hour discrete time interval (up to 72 hours) |
| `age` | PATIENTS / ADMISSIONS | Years | Static | Patient age at ICU admission (clipped >= 18) |
| `gender` | PATIENTS.csv | Binary | 0/1 | Biological sex (0: Female, 1: Male) |
| `elixhauser` | DIAGNOSES_ICD.csv | Integer | 0..10 | Premorbid comorbidity score |
| `sofa_initial` | Calculated | Integer | 2..18 | Sequential Organ Failure Assessment at baseline |

---

## 2. Vital Signs (4-Hour Temporal Resolution)

| Variable Name | Standard Unit | Normal Range | Aggregation | Physiological Implication |
|---|---|---|---|---|
| `hr` | bpm | 60 - 100 | Mean over 4h | Tachycardia / Septic response |
| `map` | mmHg | 70 - 105 | Mean over 4h | Perfusion pressure target in shock (target >= 65) |
| `spo2` | % | 95 - 100 | Mean over 4h | Peripheral oxygen saturation |
| `temp` | °C | 36.5 - 37.5 | Mean over 4h | Core body temperature / SIRS response |
| `rr` | breaths/min | 12 - 20 | Mean over 4h | Tachypnea / Respiratory distress |

---

## 3. Laboratory Biomarkers and Organ Function

| Variable Name | Standard Unit | Normal Range | Aggregation | Organ System |
|---|---|---|---|---|
| `lactate` | mmol/L | 0.5 - 2.0 | Max over 4h | Tissue hypoperfusion / Cellular hypoxia |
| `creatinine` | mg/dL | 0.6 - 1.2 | Max over 4h | Renal dysfunction (SOFA component) |
| `platelets` | 10^9/L | 150 - 400 | Min over 4h | Coagulopathy / Thrombocytopenia |
| `bilirubin` | mg/dL | 0.2 - 1.2 | Max over 4h | Hepatic dysfunction |
| `wbc` | 10^9/L | 4.0 - 11.0 | Mean over 4h | Leukocytosis / Leukopenia (Infection marker) |
| `urine_output` | mL | 120 - 300 | Sum over 4h | End-organ renal perfusion |

---

## 4. Medical Interventions and Action Space

The action space consists of a 5 x 5 discrete grid (25 total action IDs, from 0 to 24), defined by:
`Action ID = Fluid Bin * 5 + Vasopressor Bin`

### 4.1. Intravenous Fluids (IV Fluids)
- Measured as total fluid volume administered over the 4-hour window (crystalloids, colloids, blood products).
- Bin 0: Exactly 0.0 mL (no fluid bolus or background infusion).
- Bin 1: (0 mL, 25th percentile of positive fluid volume on Train].
- Bin 2: (25th percentile, 50th percentile of positive fluid volume on Train].
- Bin 3: (50th percentile, 75th percentile of positive fluid volume on Train].
- Bin 4: > 75th percentile of positive fluid volume on Train.

### 4.2. Vasopressors
- Measured as the maximum infusion rate over the 4-hour window, normalized to norepinephrine-equivalent dose (mcg/kg/min).
- Bin 0: Exactly 0.0 mcg/kg/min (no vasopressor administered).
- Bin 1: (0.0, 25th percentile of positive vasopressor rate on Train].
- Bin 2: (25th percentile, 50th percentile of positive vasopressor rate on Train].
- Bin 3: (50th percentile, 75th percentile of positive vasopressor rate on Train].
- Bin 4: > 75th percentile of positive vasopressor rate on Train.

---

## 5. MDP State Space and Rewards

- Health States: 75 discrete clusters (0 to 74) derived from K-means++ clustering on scaled patient observations.
- Terminal Absorbing States:
  - State 75: Alive / Discharged alive.
  - State 76: Dead / Mortality within 90 days.
- Reward Function:
  - Intermediate transitions (s -> s'): Reward = 0.0.
  - Transition into State 75 (Alive): Reward = +100.0.
  - Transition into State 76 (Dead): Reward = -100.0.
- Discount Factor: gamma = 0.99.
