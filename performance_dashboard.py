"""
Performance Metrics Dashboard for Traffic Optimization

Organizes and presents optimization results in a structured, publication-ready format.
Provides comprehensive metrics, visualizations, and analysis tools.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime


class PerformanceDashboard:
    """
    Comprehensive dashboard for traffic optimization performance metrics.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_scenario_summary(self, scenario_id: str, algorithm: str,
                               results: Dict, network_info: Dict = None) -> Dict:
        """
        Create comprehensive summary for a single scenario.
        """
        summary = {
            'scenario_id': scenario_id,
            'algorithm': algorithm,
            'timestamp': self.timestamp,
            'performance_metrics': {},
            'optimization_details': {},
            'network_info': network_info or {}
        }

        performance_metrics = [
            'avg_waiting_time', 'max_queue_length', 'blocked_intersections',
            'total_vehicles', 'computation_time', 'objective_value'
        ]

        for metric in performance_metrics:
            if metric in results:
                summary['performance_metrics'][metric] = results[metric]

        optimization_details = [
            'iterations', 'convergence_iteration', 'best_objective_history',
            'final_solution', 'improvement_percentage'
        ]

        for detail in optimization_details:
            if detail in results:
                summary['optimization_details'][detail] = results[detail]

        return summary

    def create_multi_scenario_dashboard(self, results_dict: Dict,
                                       scenario_names: Dict = None) -> Dict:
        """
        Create dashboard for multiple scenarios and algorithms.
        """
        print(f"{'~' * 5} CREATING MULTI-SCENARIO DASHBOARD {'~' * 5}")

        dashboard = {
            'timestamp': self.timestamp,
            'scenarios': {},
            'overall_summary': {}
        }

        for scenario_id, algorithm_results in results_dict.items():
            scenario_name = scenario_names.get(scenario_id, f"Scenario {scenario_id}") if scenario_names else f"Scenario {scenario_id}"

            dashboard['scenarios'][scenario_id] = {
                'name': scenario_name,
                'algorithms': {}
            }

            for algorithm, results in algorithm_results.items():
                summary = self.create_scenario_summary(scenario_id, algorithm, results)
                dashboard['scenarios'][scenario_id]['algorithms'][algorithm] = summary

        dashboard['overall_summary'] = self._calculate_overall_summary(dashboard['scenarios'])
        print(f"[*] Dashboard created with {len(dashboard['scenarios'])} scenarios")
        print("~" * 60)

        return dashboard

    def _calculate_overall_summary(self, scenarios_data: Dict) -> Dict:
        """
        Calculate overall summary across all scenarios.
        """
        overall = {
            'total_scenarios': len(scenarios_data),
            'algorithms_analyzed': set(),
            'performance_aggregates': {}
        }

        for scenario_id, scenario_data in scenarios_data.items():
            for algorithm in scenario_data['algorithms'].keys():
                overall['algorithms_analyzed'].add(algorithm)

        overall['algorithms_analyzed'] = list(overall['algorithms_analyzed'])

        for algorithm in overall['algorithms_analyzed']:
            waiting_times = []
            queue_lengths = []
            blocked_counts = []
            computation_times = []

            for scenario_id, scenario_data in scenarios_data.items():
                if algorithm in scenario_data['algorithms']:
                    metrics = scenario_data['algorithms'][algorithm]['performance_metrics']

                    if 'avg_waiting_time' in metrics:
                        waiting_times.append(metrics['avg_waiting_time'])
                    if 'max_queue_length' in metrics:
                        queue_lengths.append(metrics['max_queue_length'])
                    if 'blocked_intersections' in metrics:
                        blocked_counts.append(metrics['blocked_intersections'])
                    if 'computation_time' in metrics:
                        computation_times.append(metrics['computation_time'])

            overall['performance_aggregates'][algorithm] = {
                'avg_waiting_time': {
                    'mean': float(np.mean(waiting_times)) if waiting_times else None,
                    'std': float(np.std(waiting_times)) if waiting_times else None,
                    'min': float(np.min(waiting_times)) if waiting_times else None,
                    'max': float(np.max(waiting_times)) if waiting_times else None,
                    'count': len(waiting_times)
                },
                'max_queue_length': {
                    'mean': float(np.mean(queue_lengths)) if queue_lengths else None,
                    'std': float(np.std(queue_lengths)) if queue_lengths else None,
                    'min': float(np.min(queue_lengths)) if queue_lengths else None,
                    'max': float(np.max(queue_lengths)) if queue_lengths else None,
                    'count': len(queue_lengths)
                },
                'blocked_intersections': {
                    'mean': float(np.mean(blocked_counts)) if blocked_counts else None,
                    'std': float(np.std(blocked_counts)) if blocked_counts else None,
                    'min': float(np.min(blocked_counts)) if blocked_counts else None,
                    'max': float(np.max(blocked_counts)) if blocked_counts else None,
                    'count': len(blocked_counts)
                },
                'computation_time': {
                    'mean': float(np.mean(computation_times)) if computation_times else None,
                    'std': float(np.std(computation_times)) if computation_times else None,
                    'min': float(np.min(computation_times)) if computation_times else None,
                    'max': float(np.max(computation_times)) if computation_times else None,
                    'count': len(computation_times)
                }
            }

        return overall

    def create_performance_overview_plot(self, dashboard_data: Dict):
        """
        Create comprehensive performance overview plot.
        """
        print(f"{'~' * 5} CREATING PERFORMANCE OVERVIEW PLOT {'~' * 5}")

        scenarios = list(dashboard_data['scenarios'].keys())
        scenario_names = [dashboard_data['scenarios'][s]['name'] for s in scenarios]
        algorithms = dashboard_data['overall_summary']['algorithms_analyzed']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        x_pos = np.arange(len(scenarios))
        width = 0.35 if len(algorithms) == 2 else 0.25

        colors = ['steelblue', 'coral', 'forestgreen', 'purple']
        for i, algorithm in enumerate(algorithms):
            waiting_times = []
            for scenario_id in scenarios:
                if algorithm in dashboard_data['scenarios'][scenario_id]['algorithms']:
                    metrics = dashboard_data['scenarios'][scenario_id]['algorithms'][algorithm]['performance_metrics']
                    waiting_times.append(metrics.get('avg_waiting_time', 0))
                else:
                    waiting_times.append(0)

            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax1.bar(x_pos + offset, waiting_times, width, label=algorithm,
                         color=colors[i % len(colors)], alpha=0.8)

            for bar, value in zip(bars, waiting_times):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        ax1.set_xlabel('Scenario', fontsize=11)
        ax1.set_ylabel('Average Waiting Time (s)', fontsize=11)
        ax1.set_title('Waiting Time Performance by Scenario', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[0, 1]
        for i, algorithm in enumerate(algorithms):
            queue_lengths = []
            for scenario_id in scenarios:
                if algorithm in dashboard_data['scenarios'][scenario_id]['algorithms']:
                    metrics = dashboard_data['scenarios'][scenario_id]['algorithms'][algorithm]['performance_metrics']
                    queue_lengths.append(metrics.get('max_queue_length', 0))
                else:
                    queue_lengths.append(0)

            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax2.bar(x_pos + offset, queue_lengths, width, label=algorithm,
                         color=colors[i % len(colors)], alpha=0.8)

            for bar, value in zip(bars, queue_lengths):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{value:.0f}', ha='center', va='bottom', fontsize=8)

        ax2.set_xlabel('Scenario', fontsize=11)
        ax2.set_ylabel('Maximum Queue Length (vehicles)', fontsize=11)
        ax2.set_title('Queue Length Performance by Scenario', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = axes[1, 0]
        for i, algorithm in enumerate(algorithms):
            blocked_counts = []
            for scenario_id in scenarios:
                if algorithm in dashboard_data['scenarios'][scenario_id]['algorithms']:
                    metrics = dashboard_data['scenarios'][scenario_id]['algorithms'][algorithm]['performance_metrics']
                    blocked_counts.append(metrics.get('blocked_intersections', 0))
                else:
                    blocked_counts.append(0)

            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax3.bar(x_pos + offset, blocked_counts, width, label=algorithm,
                         color=colors[i % len(colors)], alpha=0.8)

            for bar, value in zip(bars, blocked_counts):
                if value > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{value:.0f}', ha='center', va='bottom', fontsize=8)

        ax3.set_xlabel('Scenario', fontsize=11)
        ax3.set_ylabel('Blocked Intersections', fontsize=11)
        ax3.set_title('Blocked Intersections by Scenario', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        for i, algorithm in enumerate(algorithms):
            comp_times = []
            for scenario_id in scenarios:
                if algorithm in dashboard_data['scenarios'][scenario_id]['algorithms']:
                    metrics = dashboard_data['scenarios'][scenario_id]['algorithms'][algorithm]['performance_metrics']
                    comp_times.append(metrics.get('computation_time', 0))
                else:
                    comp_times.append(0)

            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax4.bar(x_pos + offset, comp_times, width, label=algorithm,
                         color=colors[i % len(colors)], alpha=0.8)

            for bar, value in zip(bars, comp_times):
                if value > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        ax4.set_xlabel('Scenario', fontsize=11)
        ax4.set_ylabel('Computation Time (s)', fontsize=11)
        ax4.set_title('Computation Time by Scenario', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'performance_overview.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[*] Performance overview plot saved to {plot_path}")
        print("~" * 60)

    def create_algorithm_comparison_matrix(self, dashboard_data: Dict):
        """
        Create algorithm comparison matrix heatmap.
        """
        print(f"{'~' * 5} CREATING ALGORITHM COMPARISON MATRIX {'~' * 5}")

        algorithms = dashboard_data['overall_summary']['algorithms_analyzed']
        scenarios = list(dashboard_data['scenarios'].keys())

        if len(algorithms) < 2:
            print("[!] Need at least 2 algorithms for comparison matrix")
            return

        metrics = ['avg_waiting_time', 'max_queue_length', 'blocked_intersections']
        metric_names = ['Avg Waiting Time (s)', 'Max Queue Length', 'Blocked Intersections']

        short_labels = []
        for scenario_id in scenarios:
            short_id = scenario_id.split('_')[0] if '_' in scenario_id else scenario_id[:3]
            short_labels.append(short_id)

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            fig, ax = plt.subplots(figsize=(8, max(12, len(scenarios) * 0.4)))

            matrix = np.zeros((len(scenarios), len(algorithms)))

            for i, scenario_id in enumerate(scenarios):
                for j, algorithm in enumerate(algorithms):
                    if algorithm in dashboard_data['scenarios'][scenario_id]['algorithms']:
                        metrics_data = dashboard_data['scenarios'][scenario_id]['algorithms'][algorithm]['performance_metrics']
                        matrix[i, j] = metrics_data.get(metric, 0)

            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')

            ax.set_xticks(np.arange(len(algorithms)))
            ax.set_yticks(np.arange(len(scenarios)))
            ax.set_xticklabels(algorithms, fontsize=11)
            ax.set_yticklabels(short_labels, fontsize=9)

            for i in range(len(scenarios)):
                for j in range(len(algorithms)):
                    value = matrix[i, j]
                    if value > 0:
                        text_color = 'white' if value > np.mean(matrix) + np.std(matrix) else 'black'
                        ax.text(j, i, f'{value:.1f}',
                               ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')

            ax.set_title(f'{metric_name}\nComparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Algorithm', fontsize=11)
            ax.set_ylabel('Scenario', fontsize=11)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_name, fontsize=10)

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'algorithm_comparison_{metric}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[*] {metric_name} comparison saved to {plot_path}")

        print("~" * 60)

    def create_executive_summary(self, dashboard_data: Dict) -> str:
        """
        Create executive summary of results.
        """
        lines = []
        lines.append(f"{'~' * 5} PERFORMANCE DASHBOARD - EXECUTIVE SUMMARY {'~' * 5}")
        lines.append(f"Generated: {self.timestamp}")
        lines.append("~" * 60)

        overall = dashboard_data['overall_summary']
        lines.append(f"\nOVERALL SUMMARY")
        lines.append(f"Total Scenarios Analyzed: {overall['total_scenarios']}")
        lines.append(f"Algorithms Analyzed: {', '.join(overall['algorithms_analyzed'])}")

        lines.append(f"\nPERFORMANCE HIGHLIGHTS")
        lines.append("~" * 60)

        for algorithm in overall['algorithms_analyzed']:
            aggregates = overall['performance_aggregates'][algorithm]

            lines.append(f"\n{algorithm} Algorithm:")
            lines.append(f"  Average Waiting Time:")
            lines.append(f"    Mean: {aggregates['avg_waiting_time']['mean']:.2f}s ± {aggregates['avg_waiting_time']['std']:.2f}s")
            lines.append(f"    Range: [{aggregates['avg_waiting_time']['min']:.1f}s, {aggregates['avg_waiting_time']['max']:.1f}s]")

            lines.append(f"  Maximum Queue Length:")
            lines.append(f"    Mean: {aggregates['max_queue_length']['mean']:.1f} ± {aggregates['max_queue_length']['std']:.1f}")
            lines.append(f"    Range: [{aggregates['max_queue_length']['min']:.0f}, {aggregates['max_queue_length']['max']:.0f}]")

            lines.append(f"  Blocked Intersections:")
            lines.append(f"    Mean: {aggregates['blocked_intersections']['mean']:.2f} ± {aggregates['blocked_intersections']['std']:.2f}")
            lines.append(f"    Range: [{aggregates['blocked_intersections']['min']:.0f}, {aggregates['blocked_intersections']['max']:.0f}]")

            lines.append(f"  Computation Time:")
            lines.append(f"    Mean: {aggregates['computation_time']['mean']:.2f}s ± {aggregates['computation_time']['std']:.2f}s")
            lines.append(f"    Range: [{aggregates['computation_time']['min']:.1f}s, {aggregates['computation_time']['max']:.1f}s]")

        lines.append("\n" + "~" * 60)
        lines.append("SCENARIO BREAKDOWN")
        lines.append("~" * 60)

        for scenario_id, scenario_data in dashboard_data['scenarios'].items():
            lines.append(f"\n{scenario_data['name']}:")
            for algorithm, algorithm_data in scenario_data['algorithms'].items():
                metrics = algorithm_data['performance_metrics']
                lines.append(f"  {algorithm}:")
                lines.append(f"    Avg Waiting Time: {metrics.get('avg_waiting_time', 'N/A'):.2f}s")
                lines.append(f"    Max Queue Length: {metrics.get('max_queue_length', 'N/A'):.1f}")
                lines.append(f"    Blocked Intersections: {metrics.get('blocked_intersections', 'N/A'):.0f}")
                lines.append(f"    Computation Time: {metrics.get('computation_time', 'N/A'):.2f}s")

        lines.append("\n" + "~" * 60)
        lines.append("END OF EXECUTIVE SUMMARY")
        lines.append("~" * 60)

        return "\n".join(lines)

    def save_dashboard(self, dashboard_data: Dict):
        """
        Save complete dashboard to files.
        """
        print(f"{'~' * 5} SAVING DASHBOARD DATA {'~' * 5}")

        json_path = os.path.join(self.output_dir, 'dashboard_data.json')
        with open(json_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        print(f"[*] Dashboard data saved to {json_path}")

        summary = self.create_executive_summary(dashboard_data)
        summary_path = os.path.join(self.output_dir, 'executive_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"[*] Executive summary saved to {summary_path}")

        self.create_performance_overview_plot(dashboard_data)
        self.create_algorithm_comparison_matrix(dashboard_data)

        print(f"[*] Dashboard generation complete!")
        print(f"[*] All outputs saved to: {self.output_dir}")
        print("~" * 60)

    def generate_dashboard_from_results(self, results_dict: Dict,
                                      scenario_names: Dict = None) -> Dict:
        """
        Generate complete dashboard from results dictionary.
        """
        print(f"{'~' * 5} GENERATING PERFORMANCE DASHBOARD {'~' * 5}")

        dashboard_data = self.create_multi_scenario_dashboard(results_dict, scenario_names)
        self.save_dashboard(dashboard_data)

        return dashboard_data


if __name__ == "__main__":
    print("Performance Metrics Dashboard")
    print("Features: multi-scenario tracking, comparison matrices, executive summaries")