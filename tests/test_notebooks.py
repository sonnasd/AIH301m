import json
import os
import sys
import unittest
import matplotlib
matplotlib.use('Agg')

class TestNotebooksExecution(unittest.TestCase):
    def run_single_notebook(self, nb_path):
        self.assertTrue(os.path.exists(nb_path), f"File {nb_path} does not exist")
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        nb_dir = os.path.dirname(os.path.abspath(nb_path))
        old_cwd = os.getcwd()
        os.chdir(nb_dir)
        
        env = {'__name__': '__main__'}
        try:
            for idx, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    code = ''.join(cell['source'])
                    exec(code, env)
        finally:
            os.chdir(old_cwd)
            
    def test_member1_notebook(self):
        self.run_single_notebook('member_1_data/member_1_data_pipeline.ipynb')
        
    def test_member2_notebook(self):
        self.run_single_notebook('member_2_rl/member_2_mdp_policy_iteration.ipynb')
        
    def test_member3_notebook(self):
        self.run_single_notebook('member_3_evaluation/member_3_ope_evaluation.ipynb')

if __name__ == '__main__':
    unittest.main()
