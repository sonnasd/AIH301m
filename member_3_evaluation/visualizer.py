import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class EvaluationVisualizer:
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_policy_comparison(self, boot_results, filename='policy_value_comparison.png'):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        
        policies = ['Clinicians', 'AI Clinician']
        medians = [boot_results['clin_median'], boot_results['wis_median']]
        lower_errors = [
            boot_results['clin_median'] - boot_results['clin_ci_lower'],
            boot_results['wis_median'] - boot_results['wis_ci_lower']
        ]
        upper_errors = [
            boot_results['clin_ci_upper'] - boot_results['clin_median'],
            boot_results['wis_ci_upper'] - boot_results['wis_median']
        ]
        
        y_err = [lower_errors, upper_errors]
        colors = ['#4A7BB0', '#D95F02']
        
        bars = ax.bar(policies, medians, yerr=y_err, capsize=8, color=colors, alpha=0.85, width=0.45, edgecolor='black', linewidth=1.2)
        
        for bar, med in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width() / 2, med / 2, f'{med:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            
        ax.set_ylabel('Estimated Policy Value (Return)', fontsize=12)
        ax.set_title('Off-Policy Evaluation on Held-Out ICU Cohort (95% CI)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(-100, 100)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_action_distributions(self, traj_df, ai_policy, filename='action_distribution.png'):
        traj_df = traj_df.copy()
        clin_actions = traj_df['action'].values
        ai_actions = np.array([ai_policy[int(s)] for s in traj_df['state']])
        
        clin_fluids = clin_actions // 5
        clin_vaso = clin_actions % 5
        ai_fluids = ai_actions // 5
        ai_vaso = ai_actions % 5
        
        bins = np.arange(5)
        width = 0.35
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        clin_f_counts = np.bincount(clin_fluids, minlength=5) / len(clin_fluids)
        ai_f_counts = np.bincount(ai_fluids, minlength=5) / len(ai_fluids)
        
        axes[0].bar(bins - width/2, clin_f_counts, width=width, label='Clinicians', color='#4A7BB0', edgecolor='black')
        axes[0].bar(bins + width/2, ai_f_counts, width=width, label='AI Clinician', color='#D95F02', edgecolor='black')
        axes[0].set_title('Intravenous Fluids Dose Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fluid Dose Quartile (0: None, 1-4: Dose Bins)', fontsize=11)
        axes[0].set_ylabel('Proportion', fontsize=11)
        axes[0].set_xticks(bins)
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)
        
        clin_v_counts = np.bincount(clin_vaso, minlength=5) / len(clin_vaso)
        ai_v_counts = np.bincount(ai_vaso, minlength=5) / len(ai_vaso)
        
        axes[1].bar(bins - width/2, clin_v_counts, width=width, label='Clinicians', color='#4A7BB0', edgecolor='black')
        axes[1].bar(bins + width/2, ai_v_counts, width=width, label='AI Clinician', color='#D95F02', edgecolor='black')
        axes[1].set_title('Vasopressors Dose Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Vasopressor Dose Quartile (0: None, 1-4: Dose Bins)', fontsize=11)
        axes[1].set_ylabel('Proportion', fontsize=11)
        axes[1].set_xticks(bins)
        axes[1].legend()
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_dose_difference_vs_mortality(self, traj_df, ai_policy, filename='dose_excess_vs_mortality.png'):
        traj_df = traj_df.copy()
        patient_diffs = []
        
        grouped = traj_df.groupby('subject_id')
        for subj_id, grp in grouped:
            c_actions = grp['action'].values
            a_actions = np.array([ai_policy[int(s)] for s in grp['state']])
            died = grp['died'].iloc[0]
            
            f_diff = np.mean((c_actions // 5) - (a_actions // 5))
            v_diff = np.mean((c_actions % 5) - (a_actions % 5))
            
            patient_diffs.append({
                'subject_id': subj_id,
                'f_diff': f_diff,
                'v_diff': v_diff,
                'died': died
            })
            
        p_df = pd.DataFrame(patient_diffs)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        p_df['f_diff_bin'] = pd.cut(p_df['f_diff'], bins=[-5, -1.0, -0.2, 0.2, 1.0, 5], labels=['<< AI', '< AI', 'Match', '> AI', '>> AI'])
        f_mort = p_df.groupby('f_diff_bin', observed=False)['died'].mean()
        
        axes[0].plot(f_mort.index.astype(str), f_mort.values * 100, marker='o', linewidth=2, color='#2B5C8F')
        axes[0].set_title('Mortality vs Fluid Dose Discrepancy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Clinician Dose relative to AI Recommendation', fontsize=11)
        axes[0].set_ylabel('Mortality Rate (%)', fontsize=11)
        axes[0].grid(True, linestyle='--', alpha=0.4)
        
        p_df['v_diff_bin'] = pd.cut(p_df['v_diff'], bins=[-5, -1.0, -0.2, 0.2, 1.0, 5], labels=['<< AI', '< AI', 'Match', '> AI', '>> AI'])
        v_mort = p_df.groupby('v_diff_bin', observed=False)['died'].mean()
        
        axes[1].plot(v_mort.index.astype(str), v_mort.values * 100, marker='s', linewidth=2, color='#C0392B')
        axes[1].set_title('Mortality vs Vasopressor Dose Discrepancy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Clinician Dose relative to AI Recommendation', fontsize=11)
        axes[1].set_ylabel('Mortality Rate (%)', fontsize=11)
        axes[1].grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
