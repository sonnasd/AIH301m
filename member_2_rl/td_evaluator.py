import numpy as np

class TemporalDifferenceEvaluator:
    def __init__(self, n_states=75, n_actions=25, gamma=0.99, alpha=0.01, n_passes=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.n_passes = n_passes
        self.alive_state = n_states
        self.dead_state = n_states + 1
        self.total_states = n_states + 2
        
        self.Q = np.zeros((self.total_states, self.n_actions), dtype=np.float64)
        self.V = np.zeros(self.total_states, dtype=np.float64)
        
    def fit(self, train_traj_df):
        for _ in range(self.n_passes):
            for _, row in train_traj_df.iterrows():
                s = int(row['state'])
                a = int(row['action'])
                s_next = int(row['next_state'])
                r = float(row['reward'])
                done = bool(row['done'])
                
                if 0 <= s < self.n_states and 0 <= a < self.n_actions:
                    if done:
                        target = r
                    else:
                        target = r + self.gamma * self.V[s_next]
                    
                    self.Q[s, a] += self.alpha * (target - self.Q[s, a])
                    self.V[s] = np.mean(self.Q[s, :])
                    
        self.V[self.alive_state] = 0.0
        self.V[self.dead_state] = 0.0
        return self
        
    def get_clinician_state_values(self):
        return self.V[:self.n_states].copy()
