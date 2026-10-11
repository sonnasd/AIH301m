import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class StateActionEncoder:
    def __init__(self, n_states=75, random_state=42):
        self.n_states = n_states
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_states, init='k-means++', n_init=10, random_state=random_state)
        self.alive_state = n_states
        self.dead_state = n_states + 1
        self.total_states = n_states + 2
        
        self.fluid_bins = None
        self.vaso_bins = None
        self.feature_cols = None
        self.is_fitted = False
        
    def fit(self, train_df, feature_cols):
        self.feature_cols = feature_cols
        X = train_df[self.feature_cols].values
        self.kmeans.fit(X)
        
        pos_fluids = train_df.loc[train_df['iv_fluids'] > 0, 'iv_fluids'].values
        if len(pos_fluids) > 0:
            f_q25 = float(np.percentile(pos_fluids, 25))
            f_q50 = float(np.percentile(pos_fluids, 50))
            f_q75 = float(np.percentile(pos_fluids, 75))
            self.fluid_bins = [0.0, f_q25, f_q50, f_q75]
        else:
            self.fluid_bins = [0.0, 100.0, 250.0, 500.0]
            
        pos_vaso = train_df.loc[train_df['vasopressors'] > 0, 'vasopressors'].values
        if len(pos_vaso) > 0:
            v_q25 = float(np.percentile(pos_vaso, 25))
            v_q50 = float(np.percentile(pos_vaso, 50))
            v_q75 = float(np.percentile(pos_vaso, 75))
            self.vaso_bins = [0.0, v_q25, v_q50, v_q75]
        else:
            self.vaso_bins = [0.0, 0.05, 0.15, 0.35]
            
        self.is_fitted = True
        return self
        
    def discretize_action(self, fluid_val, vaso_val):
        if fluid_val <= 0.0:
            f_bin = 0
        elif fluid_val <= self.fluid_bins[1]:
            f_bin = 1
        elif fluid_val <= self.fluid_bins[2]:
            f_bin = 2
        elif fluid_val <= self.fluid_bins[3]:
            f_bin = 3
        else:
            f_bin = 4
            
        if vaso_val <= 0.0:
            v_bin = 0
        elif vaso_val <= self.vaso_bins[1]:
            v_bin = 1
        elif vaso_val <= self.vaso_bins[2]:
            v_bin = 2
        elif vaso_val <= self.vaso_bins[3]:
            v_bin = 3
        else:
            v_bin = 4
            
        action_id = f_bin * 5 + v_bin
        return action_id, f_bin, v_bin
        
    def transform_states(self, df):
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")
        X = df[self.feature_cols].values
        return self.kmeans.predict(X)
        
    def build_trajectories(self, df, split_name='train'):
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before building trajectories")
            
        df = df.copy()
        df['state'] = self.transform_states(df)
        
        actions = []
        fluid_bins = []
        vaso_bins = []
        for f, v in zip(df['iv_fluids'], df['vasopressors']):
            a, fb, vb = self.discretize_action(f, v)
            actions.append(a)
            fluid_bins.append(fb)
            vaso_bins.append(vb)
            
        df['action'] = actions
        df['fluid_bin'] = fluid_bins
        df['vaso_bin'] = vaso_bins
        
        trajectories = []
        grouped = df.groupby('subject_id', sort=False)
        
        for subject_id, group in grouped:
            group = group.sort_values('step').reset_index(drop=True)
            n_rows = len(group)
            died = int(group['died'].iloc[0])
            hadm_id = int(group['hadm_id'].iloc[0]) if 'hadm_id' in group.columns else 150000 + int(subject_id)
            icustay_id = int(group['icustay_id'].iloc[0])
            outcome_type = 'died' if died == 1 else 'survived'
            
            for idx in range(n_rows):
                cur_step = int(group['step'].iloc[idx])
                s = int(group['state'].iloc[idx])
                a = int(group['action'].iloc[idx])
                f_val = float(group['iv_fluids'].iloc[idx])
                v_val = float(group['vasopressors'].iloc[idx])
                
                if idx < n_rows - 1:
                    s_next = int(group['state'].iloc[idx + 1])
                    r = 0.0
                    done = False
                else:
                    if died == 1:
                        s_next = self.dead_state
                        r = -100.0
                    else:
                        s_next = self.alive_state
                        r = 100.0
                    done = True
                    
                trajectories.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'icustay_id': icustay_id,
                    'time_step': cur_step,
                    'state': s,
                    'action': a,
                    'iv_fluids': f_val,
                    'vasopressors': v_val,
                    'next_state': s_next,
                    'reward': r,
                    'done': done,
                    'outcome_type': outcome_type,
                    'dataset_split': split_name,
                    'died': died
                })
                
        return pd.DataFrame(trajectories)
        
    def export_action_bins(self, filepath):
        info = {
            'fluid_bins': self.fluid_bins,
            'vaso_bins': self.vaso_bins,
            'n_states': self.n_states,
            'alive_state': self.alive_state,
            'dead_state': self.dead_state,
            'total_states': self.total_states
        }
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
            
    def export_manifest(self, filepath, n_patients_total, n_train, n_val, n_test):
        manifest = {
            'cohort_version': '1.4',
            'consensus_definition': 'Sepsis-3 (Singer et al., 2016)',
            'inclusion': [
                'Age >= 18 at ICU admission',
                'Suspected infection (blood culture and antibiotics)',
                'Acute organ dysfunction (SOFA >= 2)'
            ],
            'exclusion': [
                'Withdrawal of life support / futile care',
                'Missing fluid administration documentation',
                'ICU stay duration < 24 hours'
            ],
            'time_series_resolution': '4-hour intervals up to 72 hours around sepsis onset',
            'counts': {
                'total_patients': n_patients_total,
                'train_patients': n_train,
                'val_patients': n_val,
                'test_patients': n_test
            }
        }
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def export_features_schema(self, filepath):
        schema = {
            'raw_features_count': len(self.feature_cols),
            'scaled_features': self.feature_cols,
            'state_clusters': self.n_states,
            'action_dimensions': {
                'iv_fluids_levels': 5,
                'vasopressors_levels': 5,
                'total_actions': 25
            }
        }
        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2)
