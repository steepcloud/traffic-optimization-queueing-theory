"""
Comparative Analysis Generator for Traffic Optimization

Creates comprehensive comparison tables and visualizations for PSO vs ACO
performance across different traffic scenarios.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os
import json


class ComparativeAnalyzer:
    """
    Generate comparative analysis between optimization algorithms.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_comparison_table(self, pso_results: Dict, aco_results: Dict,
                               scenario_names: Dict = None) -> pd.DataFrame:
        """
        Create comprehensive comparison table for PSO vs ACO.
        """
        print(f"{'~' * 5} CREATING COMPARISON TABLE {'~' * 5}")

        scenarios = sorted(set(pso_results.keys()) | set(aco_results.keys()))
        comparison_data = []

        for scenario in scenarios:
            if scenario not in pso_results or scenario not in aco_results:
                continue

            pso = pso_results[scenario]
            aco = aco_results[scenario]
            name = scenario_names.get(scenario, f"Scenario {scenario}") if scenario_names else f"Scenario {scenario}"

            pso_wait = pso.get('avg_waiting_time', float('nan'))
            aco_wait = aco.get('avg_waiting_time', float('nan'))
            pso_queue = pso.get('max_queue_length', float('nan'))
            aco_queue = aco.get('max_queue_length', float('nan'))
            pso_blocked = pso.get('blocked_intersections', float('nan'))
            aco_blocked = aco.get('blocked_intersections', float('nan'))
            pso_time = pso.get('computation_time', float('nan'))
            aco_time = aco.get('computation_time', float('nan'))

            wait_improvement = ((pso_wait - aco_wait) / pso_wait * 100) if pso_wait > 0 and not np.isnan(pso_wait) else 0
            queue_improvement = ((pso_queue - aco_queue) / pso_queue * 100) if pso_queue > 0 and not np.isnan(pso_queue) else 0
            blocked_improvement = ((pso_blocked - aco_blocked) / pso_blocked * 100) if pso_blocked > 0 and not np.isnan(pso_blocked) else 0
            time_efficiency = ((aco_time - pso_time) / aco_time * 100) if aco_time > 0 and not np.isnan(aco_time) else 0

            wait_winner = "ACO" if aco_wait < pso_wait else "PSO" if pso_wait < aco_wait else "TIE"
            queue_winner = "ACO" if aco_queue < pso_queue else "PSO" if pso_queue < aco_queue else "TIE"
            blocked_winner = "ACO" if aco_blocked < pso_blocked else "PSO" if pso_blocked < aco_blocked else "TIE"
            time_winner = "PSO" if pso_time < aco_time else "ACO" if aco_time < pso_time else "TIE"

            comparison_data.append({
                'Scenario': name,
                'PSO_Wait_Time': pso_wait,
                'ACO_Wait_Time': aco_wait,
                'Wait_Improvement_%': wait_improvement,
                'Wait_Winner': wait_winner,
                'PSO_Max_Queue': pso_queue,
                'ACO_Max_Queue': aco_queue,
                'Queue_Improvement_%': queue_improvement,
                'Queue_Winner': queue_winner,
                'PSO_Blocked': pso_blocked,
                'ACO_Blocked': aco_blocked,
                'Blocked_Improvement_%': blocked_improvement,
                'Blocked_Winner': blocked_winner,
                'PSO_Comp_Time': pso_time,
                'ACO_Comp_Time': aco_time,
                'Time_Efficiency_%': time_efficiency,
                'Time_Winner': time_winner
            })

        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(self.output_dir, 'comparison_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"[*] Comparison table saved to {csv_path}")
        print("~" * 60)

        return df

    def create_summary_statistics(self, comparison_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics across all scenarios.
        """
        print(f"{'~' * 5} CALCULATING SUMMARY STATISTICS {'~' * 5}")

        summary = {}

        valid_wait = comparison_df['Wait_Improvement_%'].dropna()
        valid_queue = comparison_df['Queue_Improvement_%'].dropna()
        valid_blocked = comparison_df['Blocked_Improvement_%'].dropna()
        valid_time = comparison_df['Time_Efficiency_%'].dropna()

        wait_winners = comparison_df['Wait_Winner'].value_counts()
        queue_winners = comparison_df['Queue_Winner'].value_counts()
        blocked_winners = comparison_df['Blocked_Winner'].value_counts()
        time_winners = comparison_df['Time_Winner'].value_counts()

        summary['waiting_time'] = {
            'mean_improvement': float(valid_wait.mean()) if len(valid_wait) > 0 else 0,
            'std_improvement': float(valid_wait.std()) if len(valid_wait) > 0 else 0,
            'median_improvement': float(valid_wait.median()) if len(valid_wait) > 0 else 0,
            'min_improvement': float(valid_wait.min()) if len(valid_wait) > 0 else 0,
            'max_improvement': float(valid_wait.max()) if len(valid_wait) > 0 else 0,
            'aco_wins': int(wait_winners.get('ACO', 0)),
            'pso_wins': int(wait_winners.get('PSO', 0)),
            'ties': int(wait_winners.get('TIE', 0))
        }

        summary['max_queue'] = {
            'mean_improvement': float(valid_queue.mean()) if len(valid_queue) > 0 else 0,
            'std_improvement': float(valid_queue.std()) if len(valid_queue) > 0 else 0,
            'median_improvement': float(valid_queue.median()) if len(valid_queue) > 0 else 0,
            'aco_wins': int(queue_winners.get('ACO', 0)),
            'pso_wins': int(queue_winners.get('PSO', 0)),
            'ties': int(queue_winners.get('TIE', 0))
        }

        summary['blocked_intersections'] = {
            'mean_improvement': float(valid_blocked.mean()) if len(valid_blocked) > 0 else 0,
            'std_improvement': float(valid_blocked.std()) if len(valid_blocked) > 0 else 0,
            'median_improvement': float(valid_blocked.median()) if len(valid_blocked) > 0 else 0,
            'aco_wins': int(blocked_winners.get('ACO', 0)),
            'pso_wins': int(blocked_winners.get('PSO', 0)),
            'ties': int(blocked_winners.get('TIE', 0))
        }

        summary['computation_time'] = {
            'mean_efficiency': float(valid_time.mean()) if len(valid_time) > 0 else 0,
            'std_efficiency': float(valid_time.std()) if len(valid_time) > 0 else 0,
            'median_efficiency': float(valid_time.median()) if len(valid_time) > 0 else 0,
            'pso_wins': int(time_winners.get('PSO', 0)),
            'aco_wins': int(time_winners.get('ACO', 0)),
            'ties': int(time_winners.get('TIE', 0))
        }

        summary['total_scenarios'] = len(comparison_df)

        summary_path = os.path.join(self.output_dir, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[*] Summary statistics saved to {summary_path}")
        print("~" * 60)

        return summary

    def create_performance_plots(self, comparison_df: pd.DataFrame):
        """
        Create comprehensive performance comparison plots.
        """
        print(f"{'~' * 5} CREATING PERFORMANCE PLOTS {'~' * 5}")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        scenarios = comparison_df['Scenario']

        ax1 = axes[0, 0]
        x_pos = np.arange(len(scenarios))
        width = 0.35

        pso_waits = comparison_df['PSO_Wait_Time'].values
        aco_waits = comparison_df['ACO_Wait_Time'].values

        bars1 = ax1.bar(x_pos - width/2, pso_waits, width, label='PSO', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x_pos + width/2, aco_waits, width, label='ACO', alpha=0.8, color='coral')

        ax1.set_xlabel('Scenario', fontsize=11)
        ax1.set_ylabel('Average Waiting Time (s)', fontsize=11)
        ax1.set_title('Waiting Time Comparison: PSO vs ACO', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        ax2 = axes[0, 1]
        pso_queues = comparison_df['PSO_Max_Queue'].values
        aco_queues = comparison_df['ACO_Max_Queue'].values

        bars1 = ax2.bar(x_pos - width/2, pso_queues, width, label='PSO', alpha=0.8, color='steelblue')
        bars2 = ax2.bar(x_pos + width/2, aco_queues, width, label='ACO', alpha=0.8, color='coral')

        ax2.set_xlabel('Scenario', fontsize=11)
        ax2.set_ylabel('Maximum Queue Length (vehicles)', fontsize=11)
        ax2.set_title('Queue Length Comparison: PSO vs ACO', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = axes[1, 0]
        pso_blocked = comparison_df['PSO_Blocked'].values
        aco_blocked = comparison_df['ACO_Blocked'].values

        bars1 = ax3.bar(x_pos - width/2, pso_blocked, width, label='PSO', alpha=0.8, color='steelblue')
        bars2 = ax3.bar(x_pos + width/2, aco_blocked, width, label='ACO', alpha=0.8, color='coral')

        ax3.set_xlabel('Scenario', fontsize=11)
        ax3.set_ylabel('Blocked Intersections', fontsize=11)
        ax3.set_title('Blocked Intersections: PSO vs ACO', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        pso_times = comparison_df['PSO_Comp_Time'].values
        aco_times = comparison_df['ACO_Comp_Time'].values

        bars1 = ax4.bar(x_pos - width/2, pso_times, width, label='PSO', alpha=0.8, color='steelblue')
        bars2 = ax4.bar(x_pos + width/2, aco_times, width, label='ACO', alpha=0.8, color='coral')

        ax4.set_xlabel('Scenario', fontsize=11)
        ax4.set_ylabel('Computation Time (s)', fontsize=11)
        ax4.set_title('Computation Time: PSO vs ACO', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[*] Performance comparison plots saved to {plot_path}")
        print("~" * 60)

    def create_improvement_plots(self, comparison_df: pd.DataFrame):
        """
        Create improvement percentage plots.
        """
        print(f"{'~' * 5} CREATING IMPROVEMENT PLOTS {'~' * 5}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        scenarios = comparison_df['Scenario']
        x_pos = np.arange(len(scenarios))

        ax1 = axes[0]
        improvements = comparison_df['Wait_Improvement_%'].values
        colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]

        bars = ax1.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Scenario', fontsize=11)
        ax1.set_ylabel('Improvement (%)', fontsize=11)
        ax1.set_title('Waiting Time Improvement (PSO vs ACO)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

        ax2 = axes[1]
        improvements = comparison_df['Queue_Improvement_%'].values
        colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]

        bars = ax2.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Scenario', fontsize=11)
        ax2.set_ylabel('Improvement (%)', fontsize=11)
        ax2.set_title('Queue Length Improvement (PSO vs ACO)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = axes[2]
        improvements = comparison_df['Blocked_Improvement_%'].values
        colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]

        bars = ax3.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Scenario', fontsize=11)
        ax3.set_ylabel('Improvement (%)', fontsize=11)
        ax3.set_title('Blocked Intersections Improvement (PSO vs ACO)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'improvement_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[*] Improvement analysis plots saved to {plot_path}")
        print("~" * 60)

    def create_winner_summary_plot(self, comparison_df: pd.DataFrame):
        """
        Create winner summary plot showing algorithm dominance.
        """
        print(f"{'~' * 5} CREATING WINNER SUMMARY PLOT {'~' * 5}")

        metrics = ['Wait_Winner', 'Queue_Winner', 'Blocked_Winner', 'Time_Winner']
        metric_names = ['Waiting Time', 'Max Queue', 'Blocked', 'Comp Time']
        winner_counts = {metric: comparison_df[metric].value_counts() for metric in metrics}

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(metrics))
        width = 0.25

        pso_wins = [winner_counts[metric].get('PSO', 0) for metric in metrics]
        aco_wins = [winner_counts[metric].get('ACO', 0) for metric in metrics]
        ties = [winner_counts[metric].get('TIE', 0) for metric in metrics]

        bars1 = ax.bar(x_pos - width, pso_wins, width, label='PSO Wins', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x_pos, aco_wins, width, label='ACO Wins', color='coral', alpha=0.8)
        bars3 = ax.bar(x_pos + width, ties, width, label='Ties', color='gray', alpha=0.8)

        ax.set_xlabel('Performance Metric', fontsize=12)
        ax.set_ylabel('Number of Scenarios Won', fontsize=12)
        ax.set_title('Algorithm Dominance Summary', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'winner_summary.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[*] Winner summary plot saved to {plot_path}")
        print("~" * 60)

    def generate_comprehensive_report(self, pso_results: Dict, aco_results: Dict,
                                    scenario_names: Dict = None) -> Dict:
        """
        Generate comprehensive comparative analysis report.
        """
        print(f"{'~' * 5} GENERATING COMPREHENSIVE REPORT {'~' * 5}")

        comparison_df = self.create_comparison_table(pso_results, aco_results, scenario_names)
        summary_stats = self.create_summary_statistics(comparison_df)
        self.create_performance_plots(comparison_df)
        self.create_improvement_plots(comparison_df)
        self.create_winner_summary_plot(comparison_df)
        report = self.generate_text_report(comparison_df, summary_stats)

        report_path = os.path.join(self.output_dir, 'comparative_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[*] Comprehensive report saved to {report_path}")
        print("~" * 60)

        return {
            'comparison_table': comparison_df.to_dict(),
            'summary_statistics': summary_stats,
            'report': report
        }

    def generate_text_report(self, comparison_df: pd.DataFrame,
                           summary_stats: Dict) -> str:
        """
        Generate formatted text report.
        """
        lines = []
        lines.append(f"{'~' * 5} COMPREHENSIVE COMPARATIVE ANALYSIS REPORT {'~' * 5}")
        lines.append("PSO vs ACO Traffic Signal Optimization")
        lines.append("~" * 60)

        lines.append("\nEXECUTIVE SUMMARY")
        lines.append("~" * 60)
        lines.append(f"Total Scenarios Analyzed: {summary_stats['total_scenarios']}")

        wait_stats = summary_stats['waiting_time']
        lines.append(f"\nWaiting Time Performance:")
        lines.append(f"  Mean Improvement: {wait_stats['mean_improvement']:.2f}%")
        lines.append(f"  Std Deviation: {wait_stats['std_improvement']:.2f}%")
        lines.append(f"  Median Improvement: {wait_stats['median_improvement']:.2f}%")
        lines.append(f"  Range: [{wait_stats['min_improvement']:.1f}%, {wait_stats['max_improvement']:.1f}%]")
        lines.append(f"  ACO Wins: {wait_stats['aco_wins']}, PSO Wins: {wait_stats['pso_wins']}, Ties: {wait_stats['ties']}")

        queue_stats = summary_stats['max_queue']
        lines.append(f"\nQueue Length Performance:")
        lines.append(f"  Mean Improvement: {queue_stats['mean_improvement']:.2f}%")
        lines.append(f"  Std Deviation: {queue_stats['std_improvement']:.2f}%")
        lines.append(f"  Median Improvement: {queue_stats['median_improvement']:.2f}%")
        lines.append(f"  ACO Wins: {queue_stats['aco_wins']}, PSO Wins: {queue_stats['pso_wins']}, Ties: {queue_stats['ties']}")

        blocked_stats = summary_stats['blocked_intersections']
        lines.append(f"\nBlocked Intersections Performance:")
        lines.append(f"  Mean Improvement: {blocked_stats['mean_improvement']:.2f}%")
        lines.append(f"  Std Deviation: {blocked_stats['std_improvement']:.2f}%")
        lines.append(f"  Median Improvement: {blocked_stats['median_improvement']:.2f}%")
        lines.append(f"  ACO Wins: {blocked_stats['aco_wins']}, PSO Wins: {blocked_stats['pso_wins']}, Ties: {blocked_stats['ties']}")

        time_stats = summary_stats['computation_time']
        lines.append(f"\nComputation Efficiency:")
        lines.append(f"  Mean Efficiency: {time_stats['mean_efficiency']:.2f}%")
        lines.append(f"  Std Deviation: {time_stats['std_efficiency']:.2f}%")
        lines.append(f"  Median Efficiency: {time_stats['median_efficiency']:.2f}%")
        lines.append(f"  PSO Wins: {time_stats['pso_wins']}, ACO Wins: {time_stats['aco_wins']}, Ties: {time_stats['ties']}")

        lines.append("\n" + "~" * 60)
        lines.append("DETAILED SCENARIO RESULTS")
        lines.append("~" * 60)

        for _, row in comparison_df.iterrows():
            lines.append(f"\n{row['Scenario']}:")
            lines.append(f"  Waiting Time: PSO={row['PSO_Wait_Time']:.2f}s, ACO={row['ACO_Wait_Time']:.2f}s "
                        f"(Improvement: {row['Wait_Improvement_%']:.1f}%, Winner: {row['Wait_Winner']})")
            lines.append(f"  Max Queue: PSO={row['PSO_Max_Queue']:.1f}, ACO={row['ACO_Max_Queue']:.1f} "
                        f"(Improvement: {row['Queue_Improvement_%']:.1f}%, Winner: {row['Queue_Winner']})")
            lines.append(f"  Blocked: PSO={row['PSO_Blocked']:.0f}, ACO={row['ACO_Blocked']:.0f} "
                        f"(Improvement: {row['Blocked_Improvement_%']:.1f}%, Winner: {row['Blocked_Winner']})")
            lines.append(f"  Comp Time: PSO={row['PSO_Comp_Time']:.2f}s, ACO={row['ACO_Comp_Time']:.2f}s "
                        f"(Efficiency: {row['Time_Efficiency_%']:.1f}%, Winner: {row['Time_Winner']})")

        lines.append("\n" + "~" * 60)
        lines.append("END OF REPORT")
        lines.append("~" * 60)

        return "\n".join(lines)


if __name__ == "__main__":
    print("Comparative Analysis Generator")
    print("Features: comparison tables, summary statistics, visualization plots")