"""
Statistical Analysis Module for Traffic Optimization Results

Provides confidence intervals, significance testing, and statistical validation
for comparing optimization algorithms (PSO vs ACO) across different scenarios.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import json
import os


class StatisticalAnalyzer:
    """
    Statistical analysis tools for traffic optimization results.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level

    def calculate_confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean using t-distribution.
        """
        if len(data) < 2:
            return (np.mean(data), np.mean(data))

        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        margin_of_error = t_critical * std_err
        return (mean - margin_of_error, mean + margin_of_error)

    def paired_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """
        Perform paired t-test between two groups.
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal length for paired t-test")

        if len(group1) < 2:
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'mean_diff': 0.0,
                'std_diff': 0.0
            }

        t_stat, p_value = stats.ttest_rel(group1, group2)
        differences = group1 - group2
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'confidence_level': self.confidence_level
        }

    def independent_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """
        Perform independent two-sample t-test.
        """
        if len(group1) < 2 or len(group2) < 2:
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'mean1': np.mean(group1) if len(group1) > 0 else 0,
                'mean2': np.mean(group2) if len(group2) > 0 else 0,
                'effect_size': 0.0
            }

        t_stat, p_value = stats.ttest_ind(group1, group2)
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) +
                              (len(group2)-1)*np.var(group2, ddof=1)) /
                             (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean1': np.mean(group1),
            'mean2': np.mean(group2),
            'effect_size': effect_size,
            'confidence_level': self.confidence_level
        }

    def anova_test(self, groups: List[np.ndarray]) -> Dict:
        """
        Perform one-way ANOVA test across multiple groups.
        """
        if len(groups) < 2:
            return {
                'f_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'num_groups': len(groups)
            }

        valid_groups = [g for g in groups if len(g) >= 2]

        if len(valid_groups) < 2:
            return {
                'f_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'num_groups': len(valid_groups)
            }

        f_stat, p_value = stats.f_oneway(*valid_groups)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'num_groups': len(valid_groups),
            'confidence_level': self.confidence_level
        }

    def calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """
        Calculate effect size metrics (Cohen's d, Hedges' g).
        """
        if len(group1) < 2 or len(group2) < 2:
            return {
                'cohens_d': 0.0,
                'hedges_g': 0.0,
                'interpretation': 'insufficient_data'
            }

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return {
                'cohens_d': 0.0,
                'hedges_g': 0.0,
                'interpretation': 'no_variance'
            }

        cohens_d = (mean1 - mean2) / pooled_std
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor

        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'interpretation': interpretation
        }

    def bootstrap_confidence_interval(self, data: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval (non-parametric).
        """
        if len(data) < 2:
            return (np.mean(data), np.mean(data))

        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)
        lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)

        return (lower, upper)


def convert_to_native_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    else:
        return obj


def analyze_optimization_results(pso_results: Dict, aco_results: Dict,
                                 output_dir: str = "results") -> Dict:
    """
    Perform comprehensive statistical analysis comparing PSO vs ACO.
    """
    print(f"{'~' * 5} STATISTICAL ANALYSIS {'~' * 5}")

    analyzer = StatisticalAnalyzer(confidence_level=0.95)

    analysis = {
        'confidence_level': 0.95,
        'scenarios': {},
        'overall_comparison': {}
    }

    scenarios = set(pso_results.keys()) | set(aco_results.keys())

    for scenario in scenarios:
        if scenario not in pso_results or scenario not in aco_results:
            continue

        pso_data = pso_results[scenario]
        aco_data = aco_results[scenario]

        metrics_to_compare = [
            'avg_waiting_time',
            'max_queue_length',
            'blocked_intersections',
            'computation_time'
        ]

        scenario_analysis = {}

        for metric in metrics_to_compare:
            if metric in pso_data and metric in aco_data:
                pso_value = pso_data[metric]
                aco_value = aco_data[metric]

                if isinstance(pso_value, (int, float)):
                    pso_values = np.array([pso_value])
                    aco_values = np.array([aco_value])
                else:
                    pso_values = np.array(pso_value) if isinstance(pso_value, list) else np.array([pso_value])
                    aco_values = np.array(aco_value) if isinstance(aco_value, list) else np.array([aco_value])

                if len(pso_values) >= 2 and len(aco_values) >= 2:
                    t_test = analyzer.independent_t_test(pso_values, aco_values)
                    effect_size = analyzer.calculate_effect_size(pso_values, aco_values)
                else:
                    t_test = {'significant': False, 'p_value': 1.0}
                    effect_size = {'interpretation': 'insufficient_data'}

                scenario_analysis[metric] = {
                    'pso_mean': float(np.mean(pso_values)),
                    'aco_mean': float(np.mean(aco_values)),
                    'pso_std': float(np.std(pso_values)) if len(pso_values) > 1 else 0.0,
                    'aco_std': float(np.std(aco_values)) if len(aco_values) > 1 else 0.0,
                    'improvement': ((pso_values.mean() - aco_values.mean()) / pso_values.mean() * 100) if pso_values.mean() > 0 else 0.0,
                    'significant': t_test['significant'],
                    'p_value': t_test.get('p_value', 1.0),
                    'effect_size': effect_size
                }

        analysis['scenarios'][scenario] = scenario_analysis

    all_pso_waiting = []
    all_aco_waiting = []

    for scenario in scenarios:
        if scenario in pso_results and scenario in aco_results:
            if 'avg_waiting_time' in pso_results[scenario]:
                all_pso_waiting.append(pso_results[scenario]['avg_waiting_time'])
            if 'avg_waiting_time' in aco_results[scenario]:
                all_aco_waiting.append(aco_results[scenario]['avg_waiting_time'])

    if len(all_pso_waiting) >= 2 and len(all_aco_waiting) >= 2:
        overall_t_test = analyzer.independent_t_test(
            np.array(all_pso_waiting),
            np.array(all_aco_waiting)
        )
        overall_effect = analyzer.calculate_effect_size(
            np.array(all_pso_waiting),
            np.array(all_aco_waiting)
        )

        analysis['overall_comparison'] = {
            'pso_mean': float(np.mean(all_pso_waiting)),
            'aco_mean': float(np.mean(all_aco_waiting)),
            'significant': overall_t_test['significant'],
            'p_value': overall_t_test['p_value'],
            'effect_size': overall_effect,
            'num_scenarios': len(all_pso_waiting)
        }

    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, 'statistical_analysis.json')
    analysis_native = convert_to_native_types(analysis)

    with open(analysis_path, 'w') as f:
        json.dump(analysis_native, f, indent=2)

    print(f"[*] Statistical analysis saved to {analysis_path}")
    print("~" * 60)

    return analysis


def format_statistical_summary(analysis: Dict) -> str:
    """
    Create a formatted summary of statistical analysis results.
    """
    lines = []
    lines.append(f"{'~' * 5} STATISTICAL ANALYSIS SUMMARY {'~' * 5}")
    lines.append(f"Confidence Level: {analysis['confidence_level']*100:.0f}%")
    lines.append("~" * 60)

    if 'overall_comparison' in analysis and analysis['overall_comparison']:
        overall = analysis['overall_comparison']
        lines.append("\nOVERALL COMPARISON (All Scenarios):")
        lines.append(f"  PSO Mean Waiting Time: {overall['pso_mean']:.2f}s")
        lines.append(f"  ACO Mean Waiting Time: {overall['aco_mean']:.2f}s")
        lines.append(f"  Statistically Significant: {'YES' if overall['significant'] else 'NO'}")
        lines.append(f"  P-value: {overall['p_value']:.4f}")
        if 'effect_size' in overall:
            effect = overall['effect_size']
            lines.append(f"  Effect Size (Cohen's d): {effect['cohens_d']:.3f} ({effect['interpretation']})")

    lines.append("\nSCENARIO-BY-SCENARIO ANALYSIS:")
    lines.append("~" * 60)

    for scenario, data in analysis['scenarios'].items():
        lines.append(f"\nScenario {scenario}:")
        for metric, results in data.items():
            if metric == 'avg_waiting_time':
                lines.append(f"  Average Waiting Time:")
                lines.append(f"    PSO: {results['pso_mean']:.2f} ± {results['pso_std']:.2f}s")
                lines.append(f"    ACO: {results['aco_mean']:.2f} ± {results['aco_std']:.2f}s")
                lines.append(f"    Improvement: {results['improvement']:.1f}%")
                lines.append(f"    Significant: {'YES' if results['significant'] else 'NO'} (p={results['p_value']:.4f})")
                if 'effect_size' in results:
                    effect = results['effect_size']
                    if 'cohens_d' in effect:
                        lines.append(f"    Effect Size: {effect['cohens_d']:.3f} ({effect['interpretation']})")

    lines.append("\n" + "~" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    print("Statistical Analysis Module")
    print("Features: confidence intervals, significance testing, effect sizes")