import numpy as np

from member_1_data.trajectory_validation import validate_trajectories


def normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    if not finite.any():
        raise ValueError("No trajectory has positive target-policy support")
    weights = np.zeros_like(log_weights)
    weights[finite] = np.exp(log_weights[finite] - log_weights[finite].max())
    return weights / weights.sum()


class WeightedImportanceSampler:
    def __init__(self, ai_policy, behavior_policy, gamma=0.99, soft_prob=0.99, weight_clip=500.0):
        self.ai_policy = np.asarray(ai_policy)
        self.behavior_policy = behavior_policy
        self.n_actions = behavior_policy.n_actions
        if not 0 <= gamma < 1 or not 0 <= soft_prob <= 1 or self.n_actions < 2:
            raise ValueError("Invalid discount, target probability or action count")
        if weight_clip is not None and (not np.isfinite(weight_clip) or weight_clip <= 0):
            raise ValueError("weight_clip must be positive and finite")
        if self.ai_policy.ndim != 1 or not len(self.ai_policy):
            raise ValueError("Policy must be a nonempty action vector")
        if not np.isin(self.ai_policy, np.arange(self.n_actions)).all():
            raise ValueError("Policy contains invalid actions")
        self.gamma = gamma
        self.soft_prob = soft_prob
        self.weight_clip = weight_clip

    def get_ai_probability(self, state, action):
        if action == self.ai_policy[state]:
            return self.soft_prob
        return (1.0 - self.soft_prob) / (self.n_actions - 1)

    def evaluate_trajectories(self, traj_df):
        episode_cols = validate_trajectories(traj_df, len(self.ai_policy), self.n_actions)
        returns, log_weights, subject_ids = [], [], []
        clipped = 0
        for _, group in traj_df.groupby(episode_cols, sort=False):
            group = group.sort_values('time_step')
            value, log_rho = 0.0, 0.0
            for t, row in enumerate(group.itertuples(index=False)):
                s, a = int(row.state), int(row.action)
                value += self.gamma ** t * row.reward
                p_ai = self.get_ai_probability(s, a)
                p_beh = self.behavior_policy.get_probability(s, a)
                if not np.isfinite(p_beh) or not 0 < p_beh <= 1:
                    raise ValueError("Observed action has invalid/zero behavior probability")
                log_rho += np.log(p_ai) - np.log(p_beh) if p_ai > 0 else -np.inf
            if self.weight_clip is not None and log_rho > np.log(self.weight_clip):
                log_rho = np.log(self.weight_clip)
                clipped += 1
            returns.append(value)
            log_weights.append(log_rho)
            subject_ids.append(group['subject_id'].iloc[0])
        returns = np.asarray(returns, dtype=float)
        log_weights = np.asarray(log_weights, dtype=float)
        weights = normalize_log_weights(log_weights)
        ess = float(1 / np.sum(weights ** 2))
        return {
            'wis_value': float(weights @ returns),
            'clinician_value': float(returns.mean()),
            'ess': ess,
            'n_patients': len(set(subject_ids)),
            'n_episodes': len(returns),
            'returns': returns,
            'weights': weights,
            'log_weights': log_weights,
            'subject_ids': subject_ids,
            'clipped_episodes': clipped,
            'low_ess': ess < max(10, 0.1 * len(returns)),
        }
