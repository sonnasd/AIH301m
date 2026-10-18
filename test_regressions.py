import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import BytesIO

import numpy as np
import pandas as pd

from main_pipeline import load_configs, prepare_data, train_model, evaluate_model
from member_1_data.synthetic_data import generate_sepsis_cohort
from member_1_data.preprocessor import ClinicalPreprocessor
from member_1_data.state_action_encoder import StateActionEncoder
from member_1_data.data_quality import run_data_quality_audit
from member_1_data.download_mimic_demo import download_demo_tables
from member_1_data.extract_mimic import extract_mimic_cohort
from member_2_rl.mdp_builder import MDPBuilder
from member_2_rl.policy_iteration import PolicyIteration
from member_2_rl.td_evaluator import TemporalDifferenceEvaluator
from member_3_evaluation.behavior_policy import BehaviorPolicyEstimator
from member_3_evaluation.off_policy_wis import WeightedImportanceSampler
from member_3_evaluation.bootstrap_evaluator import BootstrapEvaluator


def episode(subject=1, stay=1, length=1, action=0, reward=100):
    return pd.DataFrame([
        dict(subject_id=subject, icustay_id=stay, time_step=t, state=0, action=action,
             next_state=1 if t == length - 1 else 0, reward=reward if t == length - 1 else 0,
             done=t == length - 1, died=int(reward < 0))
        for t in range(length)
    ])


class RegressionTests(unittest.TestCase):
    def sampler(self, soft_prob=0.99, weight_clip=None):
        behavior = BehaviorPolicyEstimator(1, 2).fit(pd.concat([episode(), episode(2, action=1)]))
        return WeightedImportanceSampler([0], behavior, soft_prob=soft_prob, weight_clip=weight_clip)

    def test_wis_preserves_tiny_relative_weights(self):
        data = pd.concat([episode(length=100, action=1), episode(2, length=101, action=1, reward=-100)])
        result = self.sampler(0.999).evaluate_trajectories(data)
        expected = (100 * 0.99 ** 99 - 0.002 * 100 * 0.99 ** 100) / 1.002
        self.assertAlmostEqual(result['wis_value'], expected)
        self.assertAlmostEqual(result['weights'].sum(), 1)

    def test_zero_target_support_is_not_smoothed(self):
        sampler = self.sampler(1)
        result = sampler.evaluate_trajectories(pd.concat([episode(), episode(2, action=1, reward=-100)]))
        self.assertEqual(result['wis_value'], 100)
        self.assertEqual(result['weights'][1], 0)
        with self.assertRaises(ValueError):
            sampler.evaluate_trajectories(episode(action=1))

    def test_zero_behavior_support_is_rejected(self):
        sampler = self.sampler()
        sampler.behavior_policy.probs[0, 0] = 0
        with self.assertRaises(ValueError):
            sampler.evaluate_trajectories(episode())

    def test_weight_clip_is_applied_before_normalization(self):
        result = self.sampler(weight_clip=0.1).evaluate_trajectories(
            pd.concat([episode(), episode(2, action=1, reward=-100)]))
        self.assertEqual(result['clipped_episodes'], 1)
        self.assertAlmostEqual(result['wis_value'], (0.1 * 100 - 0.02 * 100) / 0.12)

    def test_patient_bootstrap_retains_all_stays(self):
        result = BootstrapEvaluator(self.sampler(), n_bootstraps=20).evaluate(
            pd.concat([episode(stay=1), episode(stay=2, reward=-100)]))
        self.assertEqual(result['n_patients'], 1)
        self.assertEqual(result['n_episodes'], 2)
        np.testing.assert_allclose(result['boot_clin'], 0)

    def test_bootstrap_renormalizes_after_dominant_patient_is_absent(self):
        data = pd.concat([episode(), episode(2, length=140, action=1, reward=-100)])
        result = BootstrapEvaluator(self.sampler(0.999), n_bootstraps=100).evaluate(data)
        self.assertEqual(result['invalid_bootstraps'], 0)
        self.assertTrue((result['boot_wis'] < 0).any())

    def test_notebook_sources_are_valid_and_unexecuted(self):
        import ast
        import nbformat
        root = Path(__file__).resolve().parent
        for path in root.glob('member_*/*.ipynb'):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source)
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs, [])

    def test_td_observed_action_not_uniform_actions(self):
        solver = TemporalDifferenceEvaluator(1, 25).fit(episode())
        self.assertAlmostEqual(solver.V[0], 100)

    def test_td_matches_empirical_bellman_solution(self):
        data = pd.concat([episode(length=2), episode(2, reward=50)])
        solver = TemporalDifferenceEvaluator(1, 2).fit(data)
        self.assertAlmostEqual(solver.V[0], 150 / (3 - 0.99))

    def test_mdp_refit_resets_counts_and_averages_rewards(self):
        data = pd.concat([episode(), episode(2, reward=50)])
        mdp = MDPBuilder(1, 2, 5).build_from_trajectories(data)
        mdp.build_from_trajectories(data)
        self.assertEqual(mdp.counts.sum(), 2)
        self.assertEqual(mdp.rewards[0, 0, 1], 75)
        self.assertEqual(mdp.fallback_states, [0])
        np.testing.assert_allclose(mdp.transitions[mdp.valid_action_mask].sum(axis=1), 1)

    def test_policy_iteration_reports_nonconvergence(self):
        mdp = MDPBuilder(1, 2, 1).build_from_trajectories(episode(length=2))
        with self.assertRaises(RuntimeError):
            PolicyIteration(mdp, max_eval_iters=1).fit()
        self.assertTrue(PolicyIteration(mdp).fit().converged)

    def test_invalid_trajectory_ids_steps_and_done(self):
        for column, value in [('state', 0.5), ('action', -1), ('reward', np.inf), ('done', False), ('time_step', -1)]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.sampler().evaluate_trajectories(episode().assign(**{column: value}))
        with self.assertRaises(ValueError):
            self.sampler().evaluate_trajectories(pd.concat([episode(), episode()]))

    def test_encoder_keeps_stays_separate(self):
        raw = generate_sepsis_cohort(3, max_steps=8)
        preprocessor = ClinicalPreprocessor()
        scaled = preprocessor.fit_transform(raw)
        encoder = StateActionEncoder(2).fit(scaled, [c + '_scaled' for c in preprocessor.feature_cols])
        first = scaled[scaled['subject_id'] == 10000].copy()
        second = first.assign(icustay_id=999999, died=1 - first['died'].iloc[0])
        trajectory = encoder.build_trajectories(pd.concat([first, second]))
        self.assertEqual(int(trajectory['done'].sum()), 2)
        self.assertEqual(trajectory.groupby('icustay_id')['died'].first().nunique(), 2)
        for dose in [np.nan, np.inf, -1]:
            with self.assertRaises(ValueError):
                encoder.discretize_action(dose, 0)
        with self.assertRaises(ValueError):
            encoder.build_trajectories(pd.concat([first, first.iloc[:1]]))

    def test_preprocessor_rejects_bad_schema_and_splits(self):
        preprocessor = ClinicalPreprocessor(['x', 'y'])
        for data in [pd.DataFrame({'x': [1]}), pd.DataFrame({'x': [1], 'y': [np.nan]}),
                     pd.DataFrame({'x': [1], 'y': [np.inf]})]:
            with self.assertRaises(ValueError):
                preprocessor.fit(data)
        with self.assertRaises(ValueError):
            preprocessor.split_patients(pd.DataFrame({'subject_id': range(10)}), train_ratio=0.9, val_ratio=0.2)

    def test_synthetic_state_is_before_current_treatment(self):
        data = generate_sepsis_cohort(4)
        np.testing.assert_allclose(data['map'], (data['sys_bp'] + 2 * data['dia_bp']) / 3)
        for _, group in data.groupby('subject_id'):
            np.testing.assert_allclose(np.diff(group['cumulated_balance']),
                                       (group['iv_fluids'] - group['urine_output']).iloc[:-1], atol=1e-10)
        pd.testing.assert_frame_equal(data, generate_sepsis_cohort(4))
        with self.assertRaises(ValueError):
            generate_sepsis_cohort(0)

    def test_quality_report_detects_reverse_order(self):
        with tempfile.TemporaryDirectory() as folder:
            raw_path, report = Path(folder) / 'raw.csv', Path(folder) / 'quality.md'
            generate_sepsis_cohort(3).iloc[::-1].to_csv(raw_path, index=False)
            run_data_quality_audit(raw_path, report)
            text = report.read_text(encoding='utf-8')
            self.assertIn('contiguous steps: FAIL', text)
            self.assertIn('split data not supplied', text)

    def test_extractor_never_silently_returns_synthetic(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'raw.csv'
            with self.assertRaises(NotImplementedError):
                extract_mimic_cohort(output_csv=target)
            self.assertFalse(target.exists())

    def test_download_failure_does_not_poison_cache(self):
        response = BytesIO(b'row_id,subject_id\n1,10000\n')
        response.headers = {'Content-Length': '999'}
        with tempfile.TemporaryDirectory() as folder:
            with patch('urllib.request.urlopen', return_value=response), self.assertRaises(OSError):
                download_demo_tables(folder, ['PATIENTS.csv'])
            self.assertEqual(list(Path(folder).iterdir()), [])
            complete = BytesIO(b'row_id,subject_id\n1,10000\n')
            complete.headers = {'Content-Length': str(len(complete.getvalue()))}
            with patch('urllib.request.urlopen', return_value=complete):
                self.assertEqual(len(download_demo_tables(folder, ['PATIENTS.csv'])), 1)
            with patch('urllib.request.urlopen') as mocked:
                self.assertEqual(len(download_demo_tables(folder, ['PATIENTS.csv'])), 1)
                mocked.assert_not_called()
            with self.assertRaises(ValueError):
                download_demo_tables(folder, ['../PATIENTS.csv'])

    def test_small_end_to_end_pipeline(self):
        configs = load_configs()
        configs['cohort']['n_patients'] = 40
        configs['model']['n_clusters'] = 3
        configs['evaluation']['bootstrap']['n_bootstraps'] = 20
        with tempfile.TemporaryDirectory() as folder:
            trajectories = prepare_data(folder, configs)
            ids = [set(part['subject_id']) for part in trajectories.values()]
            self.assertTrue(all(not left & right for i, left in enumerate(ids) for right in ids[i + 1:]))
            trainer = train_model(folder, configs)
            self.assertTrue(trainer.pi.converged)
            results = evaluate_model(folder, configs)
            self.assertTrue(np.isfinite(results['point_wis']))
            self.assertGreaterEqual(results['ess'], 1)
            metrics = json.loads((Path(folder) / 'artifacts/evaluation_metrics.json').read_text())
            self.assertEqual(metrics['data_source'], 'synthetic')
            with self.assertRaises(FileExistsError):
                prepare_data(folder, configs)


if __name__ == '__main__':
    unittest.main()
