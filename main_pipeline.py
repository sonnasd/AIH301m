import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

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

ROOT = Path(__file__).resolve().parent


def load_configs():
    configs = {}
    for name in ['cohort', 'model', 'evaluation']:
        with (ROOT / 'configs' / f'{name}.yaml').open(encoding='utf-8') as handle:
            configs[name] = yaml.safe_load(handle)
    cohort, model, evaluation = (configs[name] for name in ['cohort', 'model', 'evaluation'])
    if abs(sum(cohort[key] for key in ['train_ratio', 'val_ratio', 'test_ratio']) - 1) > 1e-10:
        raise ValueError("Split ratios must sum to one")
    if cohort['max_window_hours'] % cohort['time_step_hours'] or cohort['time_step_hours'] != 4:
        raise ValueError("Synthetic generator supports integer 4-hour steps only")
    if cohort['min_age'] != 18 or cohort['imputation_strategy'] != 'median' or cohort['scaling_strategy'] != 'standard_scaler':
        raise ValueError("Unsupported cohort preprocessing configuration")
    if model['action_space'] != dict(n_actions=25, fluid_bins=5, vaso_bins=5):
        raise ValueError("Encoder supports a fixed 5x5 action space")
    if [model[key] for key in ['reward_alive', 'reward_dead', 'reward_step']] != [100, -100, 0]:
        raise ValueError("Encoder supports terminal rewards +100/-100 and zero intermediate rewards")
    if evaluation['wis']['gamma'] != model['gamma']:
        raise ValueError("Training and evaluation discounts must match")
    if evaluation['bootstrap']['resample_level'] != 'patient':
        raise ValueError("Bootstrap must resample patients")
    if evaluation['behavior_policy']['smoothing_method'] != 'Laplace' or evaluation['target_policy']['mode'] != 'soft':
        raise ValueError("Unsupported policy probability model")
    if abs(evaluation['target_policy']['recommended_prob'] + evaluation['target_policy']['off_policy_prob'] - 1) > 1e-10:
        raise ValueError("Target-policy probabilities must sum to one")
    return configs


def prepare_data(output_dir, configs=None, overwrite=False):
    output = Path(output_dir).resolve()
    configs = load_configs() if configs is None else configs
    cohort, model = configs['cohort'], configs['model']
    for folder in ['data', 'artifacts', 'figures', 'docs']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    raw_path = output / 'data/raw_sepsis_cohort.csv'
    if raw_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {raw_path}; choose a new output directory or --overwrite")
    raw = generate_sepsis_cohort(
        n_patients=cohort['n_patients'],
        max_steps=cohort['max_window_hours'] // cohort['time_step_hours'],
        random_state=cohort['random_state'],
    )
    raw.to_csv(raw_path, index=False)
    preprocessor = ClinicalPreprocessor(feature_cols=cohort['features'])
    splits = preprocessor.split_patients(raw, cohort['train_ratio'], cohort['val_ratio'], cohort['random_state'])
    run_data_quality_audit(raw_path, output / 'docs/data_quality_report.md', splits=splits, data_source='synthetic coursework demo')
    scaled = [preprocessor.fit_transform(splits[0])] + [preprocessor.transform(part) for part in splits[1:]]
    encoder = StateActionEncoder(model['n_clusters'], model['random_state'])
    encoder.fit(scaled[0], [f'{col}_scaled' for col in preprocessor.feature_cols])
    encoder.export_action_bins(output / 'artifacts/action_bins.json')
    encoder.export_manifest(output / 'artifacts/cohort_manifest.json', raw['subject_id'].nunique(),
                            *[part['subject_id'].nunique() for part in splits],
                            generation=cohort)
    encoder.export_features_schema(output / 'artifacts/features_schema.json')
    joblib.dump(preprocessor, output / 'artifacts/preprocessor.joblib')
    joblib.dump(encoder, output / 'artifacts/state_action_encoder.joblib')
    trajectories = {}
    for name, part in zip(['train', 'val', 'test'], scaled):
        trajectories[name] = encoder.build_trajectories(part, split_name=name)
        trajectories[name].to_csv(output / f'data/trajectories_{name}.csv', index=False)
    with (output / 'artifacts/run_config.json').open('w', encoding='utf-8') as handle:
        json.dump(configs, handle, indent=2)
    return trajectories


def train_model(output_dir, configs=None):
    output = Path(output_dir)
    with (output / 'artifacts/run_config.json').open(encoding='utf-8') as handle:
        fitted_configs = json.load(handle)
    if configs is not None and configs != fitted_configs:
        raise ValueError("Training configuration differs from prepared data")
    configs = fitted_configs
    model = configs['model']
    train = pd.read_csv(output / 'data/trajectories_train.csv')
    trainer = RLTrainer(
        n_states=model['n_clusters'], n_actions=model['action_space']['n_actions'],
        min_action_count=model['min_transition_count'], gamma=model['gamma'],
        theta=float(model['theta']), max_eval_iters=model['max_eval_iters'], max_pi_iters=model['max_pi_iters'],
    ).train(train)
    trainer.export(output / 'artifacts/ai_policy.json', output / 'artifacts/mdp_matrices.npz')
    return trainer


def evaluate_model(output_dir, configs=None):
    output = Path(output_dir)
    with (output / 'artifacts/run_config.json').open(encoding='utf-8') as handle:
        fitted_configs = json.load(handle)
    if configs is not None and configs != fitted_configs:
        raise ValueError("Evaluation configuration differs from the fitted run")
    configs = fitted_configs
    model, evaluation = configs['model'], configs['evaluation']
    trajectories = {name: pd.read_csv(output / f'data/trajectories_{name}.csv') for name in ['train', 'val', 'test']}
    with (output / 'artifacts/ai_policy.json').open() as handle:
        policy = json.load(handle)['policy']
    behavior = BehaviorPolicyEstimator(
        model['n_clusters'], model['action_space']['n_actions'], evaluation['behavior_policy']['alpha']
    ).fit(trajectories['train'])
    joblib.dump(behavior, output / 'artifacts/behavior_policy.joblib')
    options = dict(gamma=model['gamma'], soft_prob=evaluation['target_policy']['recommended_prob'],
                   weight_clip=evaluation['wis']['weight_clip'])
    sensitivity = SensitivityAnalyzer(
        policy, trajectories['train'], trajectories['val'], alpha=evaluation['behavior_policy']['alpha'],
        n_actions=model['action_space']['n_actions'], **options
    ).run_full_sensitivity_suite(**evaluation['sensitivity'])
    sensitivity.to_csv(output / 'data/sensitivity_analysis.csv', index=False)
    sampler = WeightedImportanceSampler(policy, behavior, **options)
    results = BootstrapEvaluator(sampler, **{
        key: evaluation['bootstrap'][key] for key in ['n_bootstraps', 'random_state', 'confidence_level']
    }).evaluate(trajectories['test'])
    metrics = {key: value for key, value in results.items() if key not in ['boot_wis', 'boot_clin']}
    metrics.update(data_source='synthetic', estimand='episode return; patient-cluster bootstrap',
                   clinical_efficacy_established=False, sensitivity_split='validation')
    with (output / 'artifacts/evaluation_metrics.json').open('w') as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    visualizer = EvaluationVisualizer(output / 'figures')
    visualizer.plot_policy_comparison(results)
    visualizer.plot_action_distributions(trajectories['test'], policy)
    visualizer.plot_dose_difference_vs_mortality(trajectories['test'], policy)
    return results


def run_pipeline(output_dir=None, overwrite=False):
    output = ROOT if output_dir is None else Path(output_dir).resolve()
    configs = load_configs()
    prepare_data(output, configs, overwrite)
    trainer = train_model(output, configs)
    results = evaluate_model(output, configs)
    print(f"Synthetic demo complete: {output}")
    print(f"Policy iteration converged: {trainer.pi.converged}; below-threshold fallback states: {len(trainer.mdp.fallback_states)}")
    print(f"Test point WIS: {results['point_wis']:.4f}; observed return: {results['point_clinician']:.4f}")
    print(f"Episode ESS: {results['ess']:.2f}/{results['n_episodes']}")
    if results['low_ess']:
        print("WARNING: Low ESS; bootstrap intervals do not establish clinical superiority.")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic AI Clinician coursework demo, not clinical evidence")
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--overwrite', action='store_true', help="Explicitly replace generated outputs in the selected directory")
    args = parser.parse_args()
    run_pipeline(args.output_dir, args.overwrite)
