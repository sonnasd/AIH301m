import numpy as np
from member_1_data.trajectory_validation import validate_trajectories

class BehaviorPolicyEstimator:
    def __init__(self, n_states=75, n_actions=25, alpha=0.05):
        if min(n_states, n_actions) < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dimensions and smoothing alpha must be positive")
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.probs = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.is_fitted = False
        
    def fit(self, train_traj_df):
        validate_trajectories(train_traj_df, self.n_states, self.n_actions)
        counts = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        for _, row in train_traj_df.iterrows():
            s = int(row['state'])
            a = int(row['action'])
            if 0 <= s < self.n_states and 0 <= a < self.n_actions:
                counts[s, a] += 1.0
                
        for s in range(self.n_states):
            smoothed_counts = counts[s] + self.alpha
            self.probs[s] = smoothed_counts / np.sum(smoothed_counts)
            
        self.is_fitted = True
        return self
        
    def get_probability(self, state, action):
        if not self.is_fitted:
            raise RuntimeError("Behavior policy must be fitted")
        if 0 <= state < self.n_states and 0 <= action < self.n_actions:
            return float(self.probs[state, action])
        raise ValueError("State/action outside behavior policy")
