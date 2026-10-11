import unittest
import numpy as np
import pandas as pd
from member_2_rl.mdp_builder import MDPBuilder
from member_2_rl.policy_iteration import PolicyIteration

class TestMember2RL(unittest.TestCase):
    def setUp(self):
        records = []
        for i in range(30):
            records.append({'subject_id': i, 'step': 0, 'state': 0, 'action': 1, 'next_state': 1, 'reward': 0.0, 'done': False})
            records.append({'subject_id': i, 'step': 1, 'state': 1, 'action': 2, 'next_state': 5, 'reward': 100.0, 'done': True})
        self.df = pd.DataFrame(records)
        self.mdp = MDPBuilder(n_states=5, n_actions=4, min_action_count=2)
        self.mdp.build_from_trajectories(self.df)
        
    def test_transition_stochasticity(self):
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                if self.mdp.valid_action_mask[s, a]:
                    prob_sum = float(np.sum(self.mdp.transitions[s, a]))
                    self.assertAlmostEqual(prob_sum, 1.0, places=5)
                    
    def test_absorbing_states(self):
        alive_s = self.mdp.alive_state
        dead_s = self.mdp.dead_state
        for a in range(self.mdp.n_actions):
            self.assertEqual(self.mdp.transitions[alive_s, a, alive_s], 1.0)
            self.assertEqual(self.mdp.transitions[dead_s, a, dead_s], 1.0)
            
    def test_policy_iteration_convergence(self):
        pi = PolicyIteration(mdp=self.mdp, gamma=0.99, theta=1e-4)
        pi.fit()
        self.assertTrue(len(pi.history) > 0)
        self.assertTrue(pi.history[-1]['stable'])

if __name__ == '__main__':
    unittest.main()
