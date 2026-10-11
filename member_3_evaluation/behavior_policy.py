import numpy as np

class BehaviorPolicyEstimator:
    def __init__(self, n_states=75, n_actions=25, alpha=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.probs = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        
    def fit(self, train_traj_df):
        counts = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        for _, row in train_traj_df.iterrows():
            s = int(row['state'])
            a = int(row['action'])
            if 0 <= s < self.n_states and 0 <= a < self.n_actions:
                counts[s, a] += 1.0
                
        for s in range(self.n_states):
            smoothed_counts = counts[s] + self.alpha
            self.probs[s] = smoothed_counts / np.sum(smoothed_counts)
            
        return self
        
    def get_probability(self, state, action):
        if 0 <= state < self.n_states and 0 <= action < self.n_actions:
            return float(self.probs[state, action])
        return 1.0 / self.n_actions
