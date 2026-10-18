import numpy as np
from member_1_data.trajectory_validation import validate_trajectories

class MDPBuilder:
    def __init__(self, n_states=75, n_actions=25, min_action_count=5):
        if min(n_states, n_actions, min_action_count) < 1:
            raise ValueError("State, action and support counts must be positive")
        self.n_states = n_states
        self.n_actions = n_actions
        self.min_action_count = min_action_count
        
        self.alive_state = n_states
        self.dead_state = n_states + 1
        self.total_states = n_states + 2
        
        self.counts = np.zeros((self.total_states, self.n_actions, self.total_states), dtype=np.float64)
        self.transitions = np.zeros((self.total_states, self.n_actions, self.total_states), dtype=np.float64)
        self.rewards = np.zeros((self.total_states, self.n_actions, self.total_states), dtype=np.float64)
        self.valid_action_mask = np.zeros((self.total_states, self.n_actions), dtype=bool)
        
    def build_from_trajectories(self, train_traj_df):
        validate_trajectories(train_traj_df, self.n_states, self.n_actions)
        self.counts.fill(0)
        self.transitions.fill(0)
        self.rewards.fill(0)
        self.valid_action_mask.fill(False)
        self.fallback_states = []
        self.unobserved_states = []
        for _, row in train_traj_df.iterrows():
            s = int(row['state'])
            a = int(row['action'])
            s_next = int(row['next_state'])
            r = float(row['reward'])
            
            if 0 <= s < self.total_states and 0 <= a < self.n_actions and 0 <= s_next < self.total_states:
                self.counts[s, a, s_next] += 1.0
                self.rewards[s, a, s_next] += r

        np.divide(self.rewards, self.counts, out=self.rewards, where=self.counts > 0)
                
        for s in range(self.n_states):
            sa_counts = self.counts[s].sum(axis=1)
            for a in range(self.n_actions):
                if sa_counts[a] >= self.min_action_count:
                    self.valid_action_mask[s, a] = True
                    self.transitions[s, a] = self.counts[s, a] / sa_counts[a]
                    
            if not np.any(self.valid_action_mask[s]):
                best_a = int(np.argmax(sa_counts))
                if sa_counts[best_a] > 0:
                    self.fallback_states.append(s)
                    self.valid_action_mask[s, best_a] = True
                    self.transitions[s, best_a] = self.counts[s, best_a] / sa_counts[best_a]
                else:
                    self.unobserved_states.append(s)
                    self.valid_action_mask[s, 0] = True
                    self.transitions[s, 0, s] = 1.0
                    
        for a in range(self.n_actions):
            self.transitions[self.alive_state, a, self.alive_state] = 1.0
            self.transitions[self.dead_state, a, self.dead_state] = 1.0
            self.valid_action_mask[self.alive_state, a] = True
            self.valid_action_mask[self.dead_state, a] = True
            
        return self
