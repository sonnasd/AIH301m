import os
import sys
import joblib

from member_1_data.synthetic_data import generate_sepsis_cohort
from member_1_data.preprocessor import ClinicalPreprocessor
from member_1_data.state_action_encoder import StateActionEncoder
from member_1_data.data_quality import run_data_quality_audit

from member_2_rl.rl_trainer import RLTrainer

from member_3_evaluation.behavior_policy import BehaviorPolicyEstimator
from member_3_evaluation.off_policy_wis import WeightedImportanceSampler
from member_3_evaluation.bootstrap_evaluator import BootstrapEvaluator
from member_3_evaluation.sensitivity import SensitivityAnalyzer
from member_3_evaluation.visualizer import EvaluationVisualizer

def run_pipeline():
    os.makedirs('data', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    raw_df = generate_sepsis_cohort(n_patients=600, random_state=42)
    raw_csv_path = 'data/raw_sepsis_cohort.csv'
    raw_df.to_csv(raw_csv_path, index=False)
    
    run_data_quality_audit(input_csv=raw_csv_path, report_md='docs/data_quality_report.md')
    
    preprocessor = ClinicalPreprocessor()
    train_raw, val_raw, test_raw = preprocessor.split_patients(raw_df, train_ratio=0.70, val_ratio=0.15, random_state=42)
    
    train_scaled = preprocessor.fit_transform(train_raw)
    val_scaled = preprocessor.transform(val_raw)
    test_scaled = preprocessor.transform(test_raw)
    
    joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
    
    scaled_feature_cols = [c + '_scaled' for c in preprocessor.feature_cols]
    encoder = StateActionEncoder(n_states=75, random_state=42)
    encoder.fit(train_scaled, scaled_feature_cols)
    encoder.export_action_bins('artifacts/action_bins.json')
    encoder.export_manifest(
        'artifacts/cohort_manifest.json',
        n_patients_total=raw_df['subject_id'].nunique(),
        n_train=train_raw['subject_id'].nunique(),
        n_val=val_raw['subject_id'].nunique(),
        n_test=test_raw['subject_id'].nunique()
    )
    encoder.export_features_schema('artifacts/features_schema.json')
    joblib.dump(encoder, 'artifacts/state_action_encoder.joblib')
    
    tr_train = encoder.build_trajectories(train_scaled, split_name='train')
    tr_val = encoder.build_trajectories(val_scaled, split_name='val')
    tr_test = encoder.build_trajectories(test_scaled, split_name='test')
    
    tr_train.to_csv('data/trajectories_train.csv', index=False)
    tr_val.to_csv('data/trajectories_val.csv', index=False)
    tr_test.to_csv('data/trajectories_test.csv', index=False)
    
    trainer = RLTrainer(n_states=75, n_actions=25, min_action_count=5, gamma=0.99)
    trainer.train(tr_train)
    trainer.export('artifacts/ai_policy.json', 'artifacts/mdp_matrices.npz')
    ai_policy = trainer.get_policy()
    
    behavior_estimator = BehaviorPolicyEstimator(n_states=75, n_actions=25, alpha=0.05)
    behavior_estimator.fit(tr_train)
    joblib.dump(behavior_estimator, 'artifacts/behavior_policy.joblib')
    
    wis_eval = WeightedImportanceSampler(
        ai_policy=ai_policy,
        behavior_policy=behavior_estimator,
        gamma=0.99,
        soft_prob=0.99,
        weight_clip=500.0
    )
    
    boot_eval = BootstrapEvaluator(wis_eval, n_bootstraps=500, random_state=42)
    boot_results = boot_eval.evaluate(tr_test)
    
    sens_analyzer = SensitivityAnalyzer(ai_policy, tr_train, tr_test, gamma=0.99)
    sens_df = sens_analyzer.run_full_sensitivity_suite()
    sens_df.to_csv('data/sensitivity_analysis.csv', index=False)
    
    visualizer = EvaluationVisualizer(output_dir='figures')
    comp_plot = visualizer.plot_policy_comparison(boot_results)
    dist_plot = visualizer.plot_action_distributions(tr_test, ai_policy)
    diff_plot = visualizer.plot_dose_difference_vs_mortality(tr_test, ai_policy)
    
    print("Pipeline completed successfully.")
    print(f"Clinician Policy Value (Bootstrap median): {boot_results['clin_median']:.2f} (95% CI: [{boot_results['clin_ci_lower']:.2f}, {boot_results['clin_ci_upper']:.2f}])")
    print(f"AI Clinician Policy Value (WIS median):    {boot_results['wis_median']:.2f} (95% CI: [{boot_results['wis_ci_lower']:.2f}, {boot_results['wis_ci_upper']:.2f}])")
    print(f"Effective Sample Size (ESS): {boot_results['ess']:.2f} / {boot_results['n_patients']}")
    print(f"Artifacts exported: artifacts/cohort_manifest.json, artifacts/features_schema.json, artifacts/ai_policy.json")
    print(f"Figures exported: {comp_plot}, {dist_plot}, {diff_plot}")

if __name__ == '__main__':
    run_pipeline()
