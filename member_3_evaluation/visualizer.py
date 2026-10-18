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
        
        policies = ['Simulated behavior', 'Soft target (WIS)']
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
        confidence = boot_results.get('confidence_level', 0.95) * 100
        ax.set_title(f'Synthetic held-out episodes ({confidence:g}% bootstrap CI)', fontsize=12, fontweight='bold')
        status = 'LOW ESS: unreliable policy comparison' if boot_results.get('low_ess', False) else 'Not evidence of clinical efficacy'
        fig.text(0.5, 0.01, f"ESS={boot_results['ess']:.2f}/{boot_results['n_episodes']} | {status}", ha='center', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(-100, 100)
        
        plt.tight_layout(rect=(0, 0.04, 1, 1))
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
        episode_cols = ['subject_id'] + [c for c in ['hadm_id', 'icustay_id'] if c in traj_df]
        records = []
        for _, group in traj_df.groupby(episode_cols, sort=False):
            observed = group['action'].to_numpy(dtype=int)
            target = np.asarray(ai_policy)[group['state'].to_numpy(dtype=int)]
            records.append({
                'f_diff': np.mean(observed // 5 - target // 5),
                'v_diff': np.mean(observed % 5 - target % 5),
                'died': group['died'].iloc[0],
            })
        episodes = pd.DataFrame(records)
        labels = ['<< AI', '< AI', 'Match', '> AI', '>> AI']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        summaries = []
        for axis, column, title in zip(axes, ['f_diff', 'v_diff'], ['Fluid', 'Vasopressor']):
            bins = pd.cut(episodes[column], [-5, -1, -0.2, 0.2, 1, 5], labels=labels)
            summary = episodes.groupby(bins, observed=False)['died'].agg(['sum', 'count'])
            n = summary['count'].to_numpy(dtype=float)
            p = summary['sum'].to_numpy(dtype=float) / np.maximum(n, 1)
            z = 1.959963984540054
            denominator = 1 + z ** 2 / np.maximum(n, 1)
            center = (p + z ** 2 / (2 * np.maximum(n, 1))) / denominator
            half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / denominator
            valid = n > 0
            x = np.arange(len(labels))
            axis.errorbar(x[valid], 100 * p[valid],
                          yerr=100 * np.array([p[valid] - (center - half)[valid], (center + half)[valid] - p[valid]]).clip(0),
                          fmt='o', capsize=5)
            axis.set_xticks(x, [f'{label}\nn={int(count)}' for label, count in zip(labels, n)])
            axis.set_title(f'{title} bin discrepancy vs simulated mortality')
            axis.set_xlabel('Mean observed minus AI bin (not physical dose)')
            axis.set_ylabel('Mortality (%)')
            axis.set_ylim(-3, 103)
            axis.grid(axis='y', linestyle='--', alpha=0.3)
            summary['mortality'] = np.where(valid, p, np.nan)
            summary['ci_lower'] = np.where(valid, center - half, np.nan)
            summary['ci_upper'] = np.where(valid, center + half, np.nan)
            summary['treatment'] = title
            summaries.append(summary.reset_index())
        fig.suptitle('Synthetic association only; not a treatment-effect estimate', fontsize=12)
        fig.text(0.5, 0.01, 'Wilson 95% intervals; episode counts shown; repeated stays may be correlated', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        pd.concat(summaries, ignore_index=True).to_csv(os.path.join(self.output_dir, 'mortality_bin_summary.csv'), index=False)
        return save_path
