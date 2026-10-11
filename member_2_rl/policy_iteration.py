import numpy as np

class PolicyIteration:
    def __init__(self, mdp, gamma=0.99, theta=1e-5, max_eval_iters=1000, max_pi_iters=100):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_eval_iters = max_eval_iters
        self.max_pi_iters = max_pi_iters
        
        self.n_states = mdp.n_states
        self.total_states = mdp.total_states
        self.n_actions = mdp.n_actions
        self.alive_state = mdp.alive_state
        self.dead_state = mdp.dead_state
        
        self.policy = np.zeros(self.total_states, dtype=int)
        for s in range(self.n_states):
            valid_actions = np.where(self.mdp.valid_action_mask[s])[0]
            if len(valid_actions) > 0:
                self.policy[s] = valid_actions[0]
            else:
                self.policy[s] = 0
                
        self.V = np.zeros(self.total_states, dtype=np.float64)
        self.history = []
        
    def evaluate_policy(self):
        for _ in range(self.max_eval_iters):
            delta = 0.0
            for s in range(self.n_states):
                a = self.policy[s]
                probs = self.mdp.transitions[s, a]
                rewards = self.mdp.rewards[s, a]
                new_v = np.sum(probs * (rewards + self.gamma * self.V))
                delta = max(delta, abs(new_v - self.V[s]))
                self.V[s] = new_v
                
            self.V[self.alive_state] = 0.0
            self.V[self.dead_state] = 0.0
            
            if delta < self.theta:
                break
                
    def improve_policy(self):
        policy_stable = True
        actions_changed = 0
        
        for s in range(self.n_states):
            old_a = self.policy[s]
            q_values = np.full(self.n_actions, -np.inf)
            valid_actions = np.where(self.mdp.valid_action_mask[s])[0]
            
            for a in valid_actions:
                probs = self.mdp.transitions[s, a]
                rewards = self.mdp.rewards[s, a]
                q_values[a] = np.sum(probs * (rewards + self.gamma * self.V))
                
            best_a = int(np.argmax(q_values))
            self.policy[s] = best_a
            
            if best_a != old_a:
                policy_stable = False
                actions_changed += 1
                
        return policy_stable, actions_changed
        
    def compute_all_q_values(self):
        Q = np.full((self.n_states, self.n_actions), -np.inf, dtype=np.float64)
        for s in range(self.n_states):
            valid_actions = np.where(self.mdp.valid_action_mask[s])[0]
            for a in valid_actions:
                probs = self.mdp.transitions[s, a]
                rewards = self.mdp.rewards[s, a]
                Q[s, a] = np.sum(probs * (rewards + self.gamma * self.V))
        return Q
        
    def fit(self):
        for it in range(self.max_pi_iters):
            self.evaluate_policy()
            stable, changed = self.improve_policy()
            mean_v = float(np.mean(self.V[:self.n_states]))
            
            self.history.append({
                'iteration': it + 1,
                'actions_changed': changed,
                'mean_v': mean_v,
                'stable': stable
            })
            
            if stable:
                break
                
        self.evaluate_policy()
        return self
