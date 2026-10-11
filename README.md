# AI Clinician: Optimal Treatment Strategies for Sepsis in Intensive Care

This project implements an end-to-end Model-based Offline Reinforcement Learning system inspired by Komorowski et al. (*Nature Medicine*, 2018) for learning individualized fluid and vasopressor strategies for sepsis patients in the ICU. The codebase is organized into three distinct, modular components designed for a three-member collaboration, complete with runnable pipelines, unit tests, and interactive Jupyter Notebooks.

---

## 1. Project Architecture and 3-Member Responsibilities

The codebase is split into three standalone folders representing each member's core engineering module:

```
ass2/
|-- member_1_data/
|   |-- download_mimic_demo.py        # Automated fetcher for PhysioNet MIMIC-III demo tables
|   |-- query_cohort.sql             # SQL definitions for Sepsis-3 criteria (Seymour et al.)
|   |-- extract_mimic.py             # ETL pipeline extracting sepsis cohort and time series
|   |-- synthetic_data.py            # High-volume EHR simulator for full pipeline execution
|   |-- preprocessor.py              # Zero-leakage patient-level splitter, imputer, scaler
|   |-- state_action_encoder.py      # K-means++ clustering (states) and 5x5 action discretization
|   |-- data_quality.py              # Automated data audit, missingness and outlier checks
|   `-- member_1_data_pipeline.ipynb # Interactive notebook for Member 1
|
|-- member_2_rl/
|   |-- mdp_builder.py               # Transition probability estimation and valid action masking
|   |-- policy_iteration.py          # Tabular Policy Iteration (Bellman expectation solver)
|   |-- rl_trainer.py                # Training orchestrator and artifact serializer
|   `-- member_2_mdp_policy_iteration.ipynb # Interactive notebook for Member 2
|
|-- member_3_evaluation/
|   |-- behavior_policy.py           # Empirical clinician policy estimator with Laplace smoothing
|   |-- off_policy_wis.py            # Trajectory-wise Weighted Importance Sampling (WIS) and ESS
|   |-- bootstrap_evaluator.py       # Patient-level bootstrap resampling (95% Confidence Intervals)
|   |-- visualizer.py                # Scientific plotting (dose distributions, policy values, mortality)
|   `-- member_3_ope_evaluation.ipynb # Interactive notebook for Member 3
|
|-- data/                            # Processed trajectories and data dictionary
|-- artifacts/                       # Serialized scalers, encoders, and trained policies
|-- figures/                         # High-resolution evaluation charts
|-- tests/                           # Unit tests for all three modules
|-- docs/                            # Data quality reports and methodological documentation
|-- main_pipeline.py                 # Unified end-to-end execution runner
|-- run_tests.py                     # Automated unit test discovery and runner
`-- requirements.txt                 # Project environment dependencies
```

---

## 2. Technical Methodology

### Module 1: Data Preprocessing and State-Action Space (Member 1)
- **Cohort Criteria**: Adult ICU patients (age >= 18) satisfying Sepsis-3 consensus (suspected infection within -24h to +72h window and SOFA score >= 2).
- **Time Window**: Discretized into 4-hour intervals covering up to 72 hours of admission.
- **Leakage Prevention**: Patient-level splitting (Train 70%, Val 15%, Test 15%). Scalers, imputers, and cluster centroids are fitted exclusively on training trajectories.
- **State Space**: 75 physiological clusters generated via K-means++ clustering on 15 vital signs and laboratory biomarkers, plus 2 terminal absorbing states (State 75: Alive, State 76: Dead).
- **Action Space**: 25 discrete actions (5 fluid dose levels x 5 vasopressor dose levels). Dose 0 indicates no medication, while non-zero doses are partitioned into 4 empirical quartiles on training observations.

### Module 2: Markov Decision Process and Policy Iteration (Member 2)
- **Transition Probabilities**: Calculated from empirical transition counts:
  `T(s, a, s') = N(s, a, s') / N(s, a)`
- **Safety Masking**: Actions observed fewer than `min_action_count` times in a given state are masked out to prevent ungrounded extrapolation.
- **Reward Structure**: `Reward = 0.0` at intermediate steps; `Reward = +100.0` upon transition into the alive state; `Reward = -100.0` upon transition into the dead state.
- **Policy Iteration**:
  1. *Policy Evaluation*: Iterative Bellman updates until maximum value change is below threshold `theta = 1e-5`:
     `V(s) = sum_{s'} T(s, pi(s), s') [ R(s, pi(s), s') + gamma * V(s') ]`
  2. *Policy Improvement*: Greedy action selection across supported action mask:
     `pi_new(s) = argmax_{a in A_valid(s)} sum_{s'} T(s, a, s') [ R(s, a, s') + gamma * V(s') ]`
  Converges to the optimal policy `pi*` within the empirical MDP.

### Module 3: Off-Policy Evaluation and Statistical Uncertainty (Member 3)
- **Clinician Behavior Policy**: Estimated from empirical action frequencies with Laplace smoothing floor to guarantee `pi_b(a|s) > 0`.
- **Soft Target Policy**: The deterministic AI policy is softened (`pi_e = 0.99` for the chosen action, remaining `0.01` distributed uniformly across other actions).
- **Weighted Importance Sampling (WIS)**:
  `rho_i = prod_t [ pi_e(a_{i,t} | s_{i,t}) / pi_b(a_{i,t} | s_{i,t}) ]`
  `V_WIS = sum_i (rho_i * G_i) / sum_i rho_i`
- **Effective Sample Size (ESS)**: Monitors importance weight degeneration:
  `ESS = (sum_i rho_i)^2 / sum_i (rho_i^2)`
- **Patient-Level Bootstrapping**: Resamples entire patient trajectories (500 iterations) to derive non-parametric 95% confidence intervals `[2.5%, 97.5%]`.

---

## 3. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
python run_tests.py
```

### Step 3: Run End-to-End Pipeline
```bash
python main_pipeline.py
```
This executes the full sequence:
1. Generates/loads Sepsis-3 patient cohort.
2. Fits preprocessing and state-action encoders.
3. Builds the MDP and solves for optimal policy via Policy Iteration.
4. Performs WIS Off-Policy Evaluation and Bootstrap uncertainty analysis on held-out test data.
5. Saves all publication figures into `figures/`.

### Step 4: Interactive Notebooks
Each member can launch and present their individual notebook:
- Member 1: `jupyter notebook member_1_data/member_1_data_pipeline.ipynb`
- Member 2: `jupyter notebook member_2_rl/member_2_mdp_policy_iteration.ipynb`
- Member 3: `jupyter notebook member_3_evaluation/member_3_ope_evaluation.ipynb`

---

## 4. Assessment Rubric Mapping

| Criterion | Code Deliverable | Slide Ownership |
|---|---|---|
| **P4** (Sepsis context & AI solution) | `synthetic_data.py`, `query_cohort.sql` | Member 1 (Slides 01-03) |
| **M4** (Feature engineering, biostatistics, data quality) | `preprocessor.py`, `data_quality.py`, `state_action_encoder.py` | Member 1 (Slides 04-06) |
| **P5** (MDP formulation & Reinforcement Learning agent) | `mdp_builder.py`, `policy_iteration.py`, `rl_trainer.py` | Member 2 (Slides 07-12) |
| **M5** (OPE evaluation, WIS, bootstrap confidence intervals) | `behavior_policy.py`, `off_policy_wis.py`, `bootstrap_evaluator.py`, `visualizer.py` | Member 3 (Slides 13-16) |
| **D2** (Critical synthesis: ML vs biostatistics, causal limits) | `data_dictionary.md`, `main_pipeline.py`, comparative figures | All Members (Slides 16-18) |

---

## 5. Academic Integrity and Clinical Disclaimers
1. **Observational EHR Data**: Retrospective observational data contains unmeasured confounding (e.g. physician clinical intuition, unrecorded vitals). High estimated policy return does not guarantee causal efficacy in actual patients.
2. **Policy Return Interpretation**: The reported policy value (e.g. 84.5 in Komorowski et al. or 93.0 in this model) represents the expected cumulative discounted return, not a survival percentage.
3. **Clinical Governance**: This software is built for educational and research purposes under the AIH301m assignment brief. It cannot be used for clinical decision support or medical prescription.
