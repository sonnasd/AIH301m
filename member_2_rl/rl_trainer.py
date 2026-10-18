import json
import numpy as np
from member_2_rl.mdp_builder import MDPBuilder
from member_2_rl.policy_iteration import PolicyIteration
from member_2_rl.td_evaluator import TemporalDifferenceEvaluator

class RLTrainer:
    def __init__(self, n_states=75, n_actions=25, min_action_count=5, gamma=0.99,
                 theta=1e-5, max_eval_iters=1000, max_pi_iters=100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.min_action_count = min_action_count
        self.gamma = gamma
        self.pi_options = dict(theta=theta, max_eval_iters=max_eval_iters, max_pi_iters=max_pi_iters)
        self.mdp = None
        self.pi = None
        self.td_eval = None
        
    def train(self, train_traj_df):
        self.mdp = MDPBuilder(
            n_states=self.n_states,
            n_actions=self.n_actions,
            min_action_count=self.min_action_count
        )
        self.mdp.build_from_trajectories(train_traj_df)
        
        self.pi = PolicyIteration(
            mdp=self.mdp,
            gamma=self.gamma,
            **self.pi_options
        )
        self.pi.fit()
        
        self.td_eval = TemporalDifferenceEvaluator(
            n_states=self.n_states,
            n_actions=self.n_actions,
            gamma=self.gamma
        )
        self.td_eval.fit(train_traj_df)
        
        return self
        
    def get_policy(self):
        if self.pi is None:
            raise RuntimeError("Trainer must be fitted before getting policy")
        return self.pi.policy[:self.n_states].copy()
        
    def export(self, policy_json_path, npz_path=None):
        policy_dict = {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'min_action_count': self.min_action_count,
            'policy': [int(a) for a in self.get_policy()],
            'mean_v': float(np.mean(self.pi.V[:self.n_states])),
            'td_clinician_mean_v': float(np.mean(self.td_eval.get_clinician_state_values())) if self.td_eval else None,
            'value_scope': 'Unweighted training-state means, not held-out policy returns',
            'clinician_method': 'Converged batch empirical-policy TD(0)',
            'converged': self.pi.converged,
            'below_threshold_fallback_states': self.mdp.fallback_states,
            'unobserved_self_loop_states': self.mdp.unobserved_states,
            'history': self.pi.history
        }
        with open(policy_json_path, 'w') as f:
            json.dump(policy_dict, f, indent=2)
            
        if npz_path:
            np.savez_compressed(
                npz_path,
                transitions=self.mdp.transitions,
                rewards=self.mdp.rewards,
                valid_mask=self.mdp.valid_action_mask,
                counts=self.mdp.counts,
                V=self.pi.V,
                td_V=self.td_eval.V if self.td_eval else None
            )
