import pandas as pd

from member_3_evaluation.behavior_policy import BehaviorPolicyEstimator
from member_3_evaluation.off_policy_wis import WeightedImportanceSampler


class SensitivityAnalyzer:
    def __init__(self, ai_policy, train_traj_df, test_traj_df, gamma=0.99,
                 alpha=0.05, soft_prob=0.99, weight_clip=500.0, n_actions=25):
        self.ai_policy = ai_policy
        self.train_traj_df = train_traj_df
        self.validation_traj_df = test_traj_df
        self.gamma = gamma
        self.alpha, self.soft_prob, self.weight_clip = alpha, soft_prob, weight_clip
        self.n_actions = n_actions

    def _evaluate(self, param_type, values):
        results = []
        for value in values:
            params = dict(alpha=self.alpha, soft_prob=self.soft_prob, weight_clip=self.weight_clip)
            params[param_type] = value
            behavior = BehaviorPolicyEstimator(len(self.ai_policy), self.n_actions, params.pop('alpha'))
            behavior.fit(self.train_traj_df)
            result = WeightedImportanceSampler(
                self.ai_policy, behavior, gamma=self.gamma, **params
            ).evaluate_trajectories(self.validation_traj_df)
            results.append({
                'split': 'validation', 'param_type': param_type, 'param_value': value,
                **{key: result[key] for key in ['wis_value', 'clinician_value', 'ess', 'clipped_episodes', 'low_ess']},
            })
        return pd.DataFrame(results)

    def evaluate_smoothing_sensitivity(self, alphas=(0.01, 0.05, 0.10, 0.20)):
        return self._evaluate('alpha', alphas)

    def evaluate_clipping_sensitivity(self, clips=(100.0, 250.0, 500.0, 1000.0)):
        return self._evaluate('weight_clip', clips)

    def evaluate_target_sensitivity(self, soft_probs=(0.95, 0.98, 0.99)):
        return self._evaluate('soft_prob', soft_probs)

    def run_full_sensitivity_suite(self, alphas=(0.01, 0.05, 0.10, 0.20),
                                   weight_clips=(100.0, 250.0, 500.0, 1000.0),
                                   soft_probs=(0.95, 0.98, 0.99)):
        return pd.concat([
            self.evaluate_smoothing_sensitivity(alphas),
            self.evaluate_clipping_sensitivity(weight_clips),
            self.evaluate_target_sensitivity(soft_probs),
        ], ignore_index=True)
