import numpy as np

from member_1_data.trajectory_validation import validate_trajectories


class TemporalDifferenceEvaluator:
    def __init__(self, n_states=75, n_actions=25, gamma=0.99, alpha=1.0, n_passes=10000):
        if not 0 <= gamma < 1 or not 0 < alpha <= 1 or n_passes < 1:
            raise ValueError("Invalid TD parameters")
        self.n_states, self.n_actions = n_states, n_actions
        self.gamma, self.alpha, self.n_passes = gamma, alpha, n_passes
        self.alive_state, self.dead_state = n_states, n_states + 1
        self.total_states = n_states + 2
        self.Q = np.zeros((self.total_states, n_actions), dtype=float)
        self.V = np.zeros(self.total_states, dtype=float)

    def fit(self, train_traj_df):
        validate_trajectories(train_traj_df, self.n_states, self.n_actions)
        self.Q.fill(0)
        self.V.fill(0)
        s = train_traj_df['state'].to_numpy(dtype=int)
        a = train_traj_df['action'].to_numpy(dtype=int)
        next_s = train_traj_df['next_state'].to_numpy(dtype=int)
        rewards = train_traj_df['reward'].to_numpy(dtype=float)
        done = train_traj_df['done'].to_numpy(dtype=bool)
        sa = s * self.n_actions + a
        counts = np.bincount(sa, minlength=self.total_states * self.n_actions).reshape(self.Q.shape)
        policy = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        observed = counts > 0
        for iteration in range(self.n_passes):
            targets = rewards + self.gamma * self.V[next_s] * ~done
            sums = np.bincount(sa, weights=targets, minlength=self.Q.size).reshape(self.Q.shape)
            updated = sums / np.maximum(counts, 1)
            self.Q[observed] += self.alpha * (updated[observed] - self.Q[observed])
            new_v = np.sum(policy * self.Q, axis=1)
            delta = np.max(np.abs(new_v - self.V))
            self.V = new_v
            if delta < 1e-8:
                self.iterations = iteration + 1
                return self
        raise RuntimeError("Batch empirical-policy TD(0) did not converge")

    def get_clinician_state_values(self):
        return self.V[:self.n_states].copy()
