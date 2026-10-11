import unittest
import numpy as np
import pandas as pd
from member_3_evaluation.behavior_policy import BehaviorPolicyEstimator
from member_3_evaluation.off_policy_wis import WeightedImportanceSampler
from member_3_evaluation.bootstrap_evaluator import BootstrapEvaluator

class TestMember3Eval(unittest.TestCase):
    def setUp(self):
        records = []
        for i in range(25):
            records.append({'subject_id': i, 'step': 0, 'state': 0, 'action': 0, 'reward': 0.0})
            records.append({'subject_id': i, 'step': 1, 'state': 1, 'action': 1, 'reward': 100.0})
        self.df = pd.DataFrame(records)
        self.beh = BehaviorPolicyEstimator(n_states=2, n_actions=4, alpha=0.1)
        self.beh.fit(self.df)
        self.ai_policy = np.array([0, 1])
        
    def test_behavior_policy_probabilities(self):
        for s in range(2):
            self.assertAlmostEqual(float(np.sum(self.beh.probs[s])), 1.0, places=5)
            self.assertTrue(all(p > 0 for p in self.beh.probs[s]))
            
    def test_wis_evaluation(self):
        wis = WeightedImportanceSampler(
            ai_policy=self.ai_policy,
            behavior_policy=self.beh,
            gamma=0.99,
            soft_prob=0.99
        )
        res = wis.evaluate_trajectories(self.df)
        self.assertIn('wis_value', res)
        self.assertIn('ess', res)
        self.assertTrue(res['ess'] > 0)
        
    def test_bootstrap_intervals(self):
        wis = WeightedImportanceSampler(
            ai_policy=self.ai_policy,
            behavior_policy=self.beh,
            gamma=0.99,
            soft_prob=0.99
        )
        boot = BootstrapEvaluator(wis, n_bootstraps=50, random_state=42)
        b_res = boot.evaluate(self.df)
        self.assertTrue(b_res['wis_ci_lower'] <= b_res['wis_median'] <= b_res['wis_ci_upper'])

if __name__ == '__main__':
    unittest.main()
