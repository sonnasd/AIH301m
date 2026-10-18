import numpy as np
from member_3_evaluation.off_policy_wis import normalize_log_weights

class BootstrapEvaluator:
    def __init__(self, wis_evaluator, n_bootstraps=500, random_state=42, confidence_level=0.95):
        if n_bootstraps < 2 or not 0 < confidence_level < 1:
            raise ValueError("Invalid bootstrap count or confidence level")
        self.wis_evaluator = wis_evaluator
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.confidence_level = confidence_level
        
    def evaluate(self, traj_df):
        rng = np.random.RandomState(self.random_state)
        eval_res = self.wis_evaluator.evaluate_trajectories(traj_df)
        
        returns = eval_res['returns']
        log_weights = eval_res['log_weights']
        subject_ids = np.asarray(eval_res['subject_ids'])
        patients = np.unique(subject_ids)
        clusters = [np.flatnonzero(subject_ids == patient) for patient in patients]
        n_samples = len(patients)
        
        boot_wis_values = []
        boot_clin_values = []
        
        for _ in range(self.n_bootstraps):
            indices = np.concatenate([clusters[i] for i in rng.choice(n_samples, size=n_samples, replace=True)])
            b_returns = returns[indices]
            try:
                b_wis = normalize_log_weights(log_weights[indices]) @ b_returns
            except ValueError:
                b_wis = np.nan
                
            b_clin = np.mean(b_returns)
            
            boot_wis_values.append(b_wis)
            boot_clin_values.append(b_clin)
            
        boot_wis_values = np.array(boot_wis_values)
        boot_clin_values = np.array(boot_clin_values)
        valid_wis = boot_wis_values[np.isfinite(boot_wis_values)]
        if not len(valid_wis):
            raise ValueError("No bootstrap replicate has target-policy support")
        lower = 100 * (1 - self.confidence_level) / 2
        
        results = {
            'point_wis': eval_res['wis_value'],
            'point_clinician': eval_res['clinician_value'],
            'ess': eval_res['ess'],
            'n_patients': n_samples,
            'n_episodes': eval_res['n_episodes'],
            'low_ess': eval_res['low_ess'],
            'clipped_episodes': eval_res['clipped_episodes'],
            'invalid_bootstraps': int(np.isnan(boot_wis_values).sum()),
            'confidence_level': self.confidence_level,
            'wis_median': float(np.median(valid_wis)),
            'wis_ci_lower': float(np.percentile(valid_wis, lower)),
            'wis_ci_upper': float(np.percentile(valid_wis, 100 - lower)),
            'clin_median': float(np.median(boot_clin_values)),
            'clin_ci_lower': float(np.percentile(boot_clin_values, lower)),
            'clin_ci_upper': float(np.percentile(boot_clin_values, 100 - lower)),
            'boot_wis': boot_wis_values,
            'boot_clin': boot_clin_values
        }
        
        return results
