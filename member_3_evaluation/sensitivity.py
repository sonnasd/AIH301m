import numpy as np
import pandas as pd
from member_3_evaluation.behavior_policy import BehaviorPolicyEstimator
from member_3_evaluation.off_policy_wis import WeightedImportanceSampler

class SensitivityAnalyzer:
    def __init__(self, ai_policy, train_traj_df, test_traj_df, gamma=0.99):
        self.ai_policy = ai_policy
        self.train_traj_df = train_traj_df
        self.test_traj_df = test_traj_df
        self.gamma = gamma
        self.n_states = len(ai_policy)
        
    def evaluate_smoothing_sensitivity(self, alphas=(0.01, 0.05, 0.10, 0.20)):
        results = []
        for a in alphas:
            beh = BehaviorPolicyEstimator(n_states=self.n_states, n_actions=25, alpha=a)
            beh.fit(self.train_traj_df)
            
            wis = WeightedImportanceSampler(
                ai_policy=self.ai_policy,
                behavior_policy=beh,
                gamma=self.gamma,
                soft_prob=0.99,
                weight_clip=500.0
            )
            res = wis.evaluate_trajectories(self.test_traj_df)
            results.append({
                'param_type': 'alpha_smoothing',
                'param_value': a,
                'wis_value': round(res['wis_value'], 2),
                'clinician_value': round(res['clinician_value'], 2),
                'ess': round(res['ess'], 2)
            })
        return pd.DataFrame(results)
        
    def evaluate_clipping_sensitivity(self, clips=(100.0, 250.0, 500.0, 1000.0)):
        beh = BehaviorPolicyEstimator(n_states=self.n_states, n_actions=25, alpha=0.05)
        beh.fit(self.train_traj_df)
        
        results = []
        for c in clips:
            wis = WeightedImportanceSampler(
                ai_policy=self.ai_policy,
                behavior_policy=beh,
                gamma=self.gamma,
                soft_prob=0.99,
                weight_clip=c
            )
            res = wis.evaluate_trajectories(self.test_traj_df)
            results.append({
                'param_type': 'weight_clip',
                'param_value': c,
                'wis_value': round(res['wis_value'], 2),
                'clinician_value': round(res['clinician_value'], 2),
                'ess': round(res['ess'], 2)
            })
        return pd.DataFrame(results)
        
    def run_full_sensitivity_suite(self):
        df_smooth = self.evaluate_smoothing_sensitivity()
        df_clip = self.evaluate_clipping_sensitivity()
        combined = pd.concat([df_smooth, df_clip], ignore_index=True)
        return combined
