import numpy as np

class WeightedImportanceSampler:
    def __init__(self, ai_policy, behavior_policy, gamma=0.99, soft_prob=0.99, weight_clip=500.0):
        self.ai_policy = ai_policy
        self.behavior_policy = behavior_policy
        self.gamma = gamma
        self.soft_prob = soft_prob
        self.weight_clip = weight_clip
        self.n_actions = 25
        
    def get_ai_probability(self, state, action):
        rec_action = self.ai_policy[state]
        if action == rec_action:
            return self.soft_prob
        return (1.0 - self.soft_prob) / (self.n_actions - 1)
        
    def evaluate_trajectories(self, traj_df):
        grouped = traj_df.groupby('subject_id', sort=False)
        
        returns = []
        weights = []
        subject_ids = []
        
        for subj_id, group in grouped:
            step_col = 'time_step' if 'time_step' in group.columns else 'step'
            group = group.sort_values(step_col)
            t_return = 0.0
            log_rho = 0.0
            
            for t, (_, row) in enumerate(group.iterrows()):
                s = int(row['state'])
                a = int(row['action'])
                r = float(row['reward'])
                
                t_return += (self.gamma ** t) * r
                
                p_ai = self.get_ai_probability(s, a)
                p_beh = self.behavior_policy.get_probability(s, a)
                
                log_rho += np.log(max(p_ai, 1e-8)) - np.log(max(p_beh, 1e-8))
                
            rho = np.exp(np.clip(log_rho, -25.0, 25.0))
            if self.weight_clip is not None:
                rho = min(rho, self.weight_clip)
                
            returns.append(t_return)
            weights.append(rho)
            subject_ids.append(subj_id)
            
        returns = np.array(returns, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            wis_value = float(np.sum(weights * returns) / sum_weights)
            ess = float((sum_weights ** 2) / np.sum(weights ** 2))
        else:
            wis_value = 0.0
            ess = 0.0
            
        clinician_value = float(np.mean(returns))
        
        return {
            'wis_value': wis_value,
            'clinician_value': clinician_value,
            'ess': ess,
            'n_patients': len(returns),
            'returns': returns,
            'weights': weights,
            'subject_ids': subject_ids
        }
