"""
Integrated Analysis Suite for Traffic Optimization Dissertation

This module integrates statistical analysis, comparative analysis, and performance
dashboarding to provide comprehensive analysis tools for publication-quality results.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
import config

from statistical_analysis import StatisticalAnalyzer, analyze_optimization_results, format_statistical_summary
from comparative_analysis import ComparativeAnalyzer
from performance_dashboard import PerformanceDashboard


class IntegratedAnalysisSuite:
    """
    Complete analysis suite for traffic optimization results.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.statistical_analyzer = StatisticalAnalyzer(confidence_level=0.95)
        self.comparative_analyzer = ComparativeAnalyzer(output_dir)
        self.performance_dashboard = PerformanceDashboard(output_dir)

    def analyze_optimization_results(self, pso_results: Dict, aco_results: Dict,
                                    scenario_names: Dict = None) -> Dict:
        """
        Perform complete analysis of optimization results.
        """
        print(f"{'~' * 5} INTEGRATED ANALYSIS SUITE {'~' * 5}")

        print("\n[1/3] Performing Statistical Analysis...")
        statistical_results = analyze_optimization_results(
            pso_results, aco_results, self.output_dir
        )

        print("\n[2/3] Performing Comparative Analysis...")
        comparative_results = self.comparative_analyzer.generate_comprehensive_report(
            pso_results, aco_results, scenario_names
        )

        print("\n[3/3] Creating Performance Dashboard...")
        results_dict = {}
        for scenario_id in set(pso_results.keys()) | set(aco_results.keys()):
            results_dict[scenario_id] = {}
            if scenario_id in pso_results:
                results_dict[scenario_id]['PSO'] = pso_results[scenario_id]
            if scenario_id in aco_results:
                results_dict[scenario_id]['ACO'] = aco_results[scenario_id]

        dashboard_results = self.performance_dashboard.generate_dashboard_from_results(
            results_dict, scenario_names
        )

        final_results = {
            'statistical_analysis': statistical_results,
            'comparative_analysis': comparative_results,
            'performance_dashboard': dashboard_results,
            'metadata': {
                'total_scenarios': len(results_dict),
                'algorithms': ['PSO', 'ACO'],
                'output_directory': self.output_dir
            }
        }

        integrated_path = os.path.join(self.output_dir, 'integrated_analysis_results.json')
        with open(integrated_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"\n[*] Integrated analysis results saved to {integrated_path}")
        print(f"{'~' * 5} ANALYSIS COMPLETE {'~' * 5}")

        return final_results

    def generate_publication_summary(self, analysis_results: Dict) -> str:
        """
        Generate publication-ready summary of all analyses.
        """
        lines = []
        lines.append(f"{'~' * 5} PUBLICATION-READY ANALYSIS SUMMARY {'~' * 5}")
        lines.append("Traffic Signal Optimization: PSO vs ACO")
        lines.append("~" * 60)

        if 'statistical_analysis' in analysis_results:
            lines.append("\nSTATISTICAL ANALYSIS SUMMARY")
            lines.append("~" * 60)
            stat_summary = format_statistical_summary(analysis_results['statistical_analysis'])
            lines.append(stat_summary)

        if 'comparative_analysis' in analysis_results:
            comp_data = analysis_results['comparative_analysis']
            if 'summary_statistics' in comp_data:
                lines.append("\nCOMPARATIVE ANALYSIS SUMMARY")
                lines.append("~" * 60)

                summary_stats = comp_data['summary_statistics']

                lines.append(f"Total Scenarios: {summary_stats['total_scenarios']}")

                wait_stats = summary_stats['waiting_time']
                lines.append(f"\nWaiting Time Performance:")
                lines.append(f"  Mean Improvement: {wait_stats['mean_improvement']:.2f}% ± {wait_stats['std_improvement']:.2f}%")
                lines.append(f"  ACO Wins: {wait_stats['aco_wins']}, PSO Wins: {wait_stats['pso_wins']}, Ties: {wait_stats['ties']}")

                queue_stats = summary_stats['max_queue']
                lines.append(f"\nQueue Length Performance:")
                lines.append(f"  Mean Improvement: {queue_stats['mean_improvement']:.2f}% ± {queue_stats['std_improvement']:.2f}%")
                lines.append(f"  ACO Wins: {queue_stats['aco_wins']}, PSO Wins: {queue_stats['pso_wins']}, Ties: {queue_stats['ties']}")

                blocked_stats = summary_stats['blocked_intersections']
                lines.append(f"\nBlocked Intersections Performance:")
                lines.append(f"  Mean Improvement: {blocked_stats['mean_improvement']:.2f}% ± {blocked_stats['std_improvement']:.2f}%")
                lines.append(f"  ACO Wins: {blocked_stats['aco_wins']}, PSO Wins: {blocked_stats['pso_wins']}, Ties: {blocked_stats['ties']}")

        if 'performance_dashboard' in analysis_results:
            dashboard = analysis_results['performance_dashboard']
            if 'overall_summary' in dashboard:
                lines.append("\nPERFORMANCE DASHBOARD SUMMARY")
                lines.append("~" * 60)

                overall = dashboard['overall_summary']
                lines.append(f"Algorithms Analyzed: {', '.join(overall['algorithms_analyzed'])}")

                for algorithm in overall['algorithms_analyzed']:
                    aggregates = overall['performance_aggregates'][algorithm]
                    lines.append(f"\n{algorithm} Overall Performance:")
                    lines.append(f"  Avg Waiting Time: {aggregates['avg_waiting_time']['mean']:.2f}s ± {aggregates['avg_waiting_time']['std']:.2f}s")
                    lines.append(f"  Max Queue: {aggregates['max_queue_length']['mean']:.1f} ± {aggregates['max_queue_length']['std']:.1f}")
                    lines.append(f"  Blocked: {aggregates['blocked_intersections']['mean']:.2f} ± {aggregates['blocked_intersections']['std']:.2f}")

        lines.append("\n" + "~" * 60)
        lines.append("END OF PUBLICATION SUMMARY")
        lines.append("~" * 60)

        return "\n".join(lines)

    def save_publication_summary(self, analysis_results: Dict):
        """
        Save publication summary to file.
        """
        summary = self.generate_publication_summary(analysis_results)
        summary_path = os.path.join(self.output_dir, 'publication_summary.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"[*] Publication summary saved to {summary_path}")


def load_results_from_directory(results_dir: str) -> tuple:
    """
    Load PSO and ACO results from directory.
    """
    pso_results = {}
    aco_results = {}

    for filename in os.listdir(results_dir):
        if filename.startswith('pso_') and filename.endswith('.json'):
            scenario_id = filename.replace('pso_', '').replace('.json', '')
            with open(os.path.join(results_dir, filename), 'r') as f:
                pso_results[scenario_id] = json.load(f)

        elif filename.startswith('aco_') and filename.endswith('.json'):
            scenario_id = filename.replace('aco_', '').replace('.json', '')
            with open(os.path.join(results_dir, filename), 'r') as f:
                aco_results[scenario_id] = json.load(f)

    return pso_results, aco_results


def create_sample_results() -> tuple:
    """
    Create sample results for testing.
    """
    scenarios = ['1A', '1B', '2A', '2B', '3A', '3B']

    pso_results = {}
    aco_results = {}

    for scenario in scenarios:
        pso_results[scenario] = {
            'avg_waiting_time': np.random.uniform(25, 45),
            'max_queue_length': np.random.uniform(8, 20),
            'blocked_intersections': np.random.randint(0, 5),
            'total_vehicles': np.random.randint(2000, 3000),
            'computation_time': np.random.uniform(30, 60),
            'objective_value': np.random.uniform(20, 50)
        }

        aco_results[scenario] = {
            'avg_waiting_time': pso_results[scenario]['avg_waiting_time'] * np.random.uniform(0.7, 0.95),
            'max_queue_length': pso_results[scenario]['max_queue_length'] * np.random.uniform(0.6, 0.9),
            'blocked_intersections': max(0, pso_results[scenario]['blocked_intersections'] - np.random.randint(0, 3)),
            'total_vehicles': pso_results[scenario]['total_vehicles'],
            'computation_time': pso_results[scenario]['computation_time'] * np.random.uniform(0.9, 1.2),
            'objective_value': pso_results[scenario]['objective_value'] * np.random.uniform(0.7, 0.9)
        }

    return pso_results, aco_results


if __name__ == "__main__":
    print("Integrated Analysis Suite for Traffic Optimization")
    print(f"{'~' * 5} DEMONSTRATION MODE {'~' * 5}")

    print("\nCreating sample results for demonstration...")
    pso_results, aco_results = create_sample_results()

    scenario_names = {
        '1A': 'Light Traffic - Baseline',
        '1B': 'Light Traffic - Optimized',
        '2A': 'Medium Traffic - Baseline',
        '2B': 'Medium Traffic - Optimized',
        '3A': 'Heavy Traffic - Baseline',
        '3B': 'Heavy Traffic - Optimized'
    }

    suite = IntegratedAnalysisSuite(output_dir="results")

    analysis_results = suite.analyze_optimization_results(
        pso_results, aco_results, scenario_names
    )

    suite.save_publication_summary(analysis_results)

    print(f"\n{'~' * 5} DEMONSTRATION COMPLETE {'~' * 5}")
    print("Check the 'results' directory for all outputs")