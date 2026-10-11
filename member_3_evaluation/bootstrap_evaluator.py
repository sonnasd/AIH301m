import numpy as np

class BootstrapEvaluator:
    def __init__(self, wis_evaluator, n_bootstraps=500, random_state=42):
        self.wis_evaluator = wis_evaluator
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        
    def evaluate(self, traj_df):
        rng = np.random.RandomState(self.random_state)
        eval_res = self.wis_evaluator.evaluate_trajectories(traj_df)
        
        returns = eval_res['returns']
        weights = eval_res['weights']
        n_samples = len(returns)
        
        boot_wis_values = []
        boot_clin_values = []
        
        for _ in range(self.n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            b_returns = returns[indices]
            b_weights = weights[indices]
            
            b_sum_w = np.sum(b_weights)
            if b_sum_w > 0:
                b_wis = np.sum(b_weights * b_returns) / b_sum_w
            else:
                b_wis = 0.0
                
            b_clin = np.mean(b_returns)
            
            boot_wis_values.append(b_wis)
            boot_clin_values.append(b_clin)
            
        boot_wis_values = np.array(boot_wis_values)
        boot_clin_values = np.array(boot_clin_values)
        
        results = {
            'point_wis': eval_res['wis_value'],
            'point_clinician': eval_res['clinician_value'],
            'ess': eval_res['ess'],
            'n_patients': n_samples,
            'wis_median': float(np.median(boot_wis_values)),
            'wis_ci_lower': float(np.percentile(boot_wis_values, 2.5)),
            'wis_ci_upper': float(np.percentile(boot_wis_values, 97.5)),
            'clin_median': float(np.median(boot_clin_values)),
            'clin_ci_lower': float(np.percentile(boot_clin_values, 2.5)),
            'clin_ci_upper': float(np.percentile(boot_clin_values, 97.5)),
            'boot_wis': boot_wis_values,
            'boot_clin': boot_clin_values
        }
        
        return results
