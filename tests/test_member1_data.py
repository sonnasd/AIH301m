import unittest
import numpy as np
import pandas as pd
from member_1_data.synthetic_data import generate_sepsis_cohort
from member_1_data.preprocessor import ClinicalPreprocessor
from member_1_data.state_action_encoder import StateActionEncoder

class TestMember1Data(unittest.TestCase):
    def setUp(self):
        self.df = generate_sepsis_cohort(n_patients=50, random_state=42)
        self.prep = ClinicalPreprocessor()
        
    def test_patient_split_no_leakage(self):
        train, val, test = self.prep.split_patients(self.df, train_ratio=0.7, val_ratio=0.15)
        s_train = set(train['subject_id'])
        s_val = set(val['subject_id'])
        s_test = set(test['subject_id'])
        self.assertEqual(len(s_train.intersection(s_val)), 0)
        self.assertEqual(len(s_train.intersection(s_test)), 0)
        self.assertEqual(len(s_val.intersection(s_test)), 0)
        
    def test_state_action_encoding(self):
        train, val, test = self.prep.split_patients(self.df, train_ratio=0.7, val_ratio=0.15)
        train_s = self.prep.fit_transform(train)
        cols = [c + '_scaled' for c in self.prep.feature_cols]
        encoder = StateActionEncoder(n_states=20, random_state=42)
        encoder.fit(train_s, cols)
        
        tr_df = encoder.build_trajectories(train_s)
        self.assertTrue(all(0 <= a < 25 for a in tr_df['action']))
        self.assertTrue(all(0 <= s < 22 for s in tr_df['state']))
        self.assertIn('reward', tr_df.columns)
        self.assertIn('done', tr_df.columns)

if __name__ == '__main__':
    unittest.main()
