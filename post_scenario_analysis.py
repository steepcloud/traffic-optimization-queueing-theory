"""
Post-Scenario Analysis Integration

This script should be run after completing scenarios with run_all_scenarios.bat
to generate comprehensive analysis of all results.
"""

import os
import re
import numpy as np
from integrated_analysis import IntegratedAnalysisSuite


def parse_log_file(log_path: str) -> dict:
    """
    Parse optimization log file to extract key metrics.
    Specifically targets the FINAL optimized results, not baseline.
    """
    if not os.path.exists(log_path):
        return None

    metrics = {
        'avg_waiting_time': None,
        'max_queue_length': None,
        'blocked_intersections': None,
        'total_vehicles': None,
        'computation_time': None,
        'objective_value': None
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        performance_sections = re.findall(r'~~~~~ PERFORMANCE METRICS ~~~~~', content)

        if len(performance_sections) >= 2:
            second_match = content.find('~~~~~ PERFORMANCE METRICS ~~~~~')
            if second_match != -1:
                second_match = content.find('~~~~~ PERFORMANCE METRICS ~~~~~', second_match + 1)
                if second_match != -1:
                    final_section = content[second_match:]

                    patterns = {
                        'avg_waiting_time': r'Average Waiting Time:\s+([\d.]+)\s+seconds',
                        'max_queue_length': r'Maximum Queue Length:\s+([\d.]+)\s+vehicles',
                        'blocked_intersections': r'Blocked Intersections:\s+([\d.]+)',
                        'total_vehicles': r'Total Vehicles Processed:\s+([\d.]+)',
                        'objective_value': r'Objective Function Value:\s+([\d.]+)'
                    }

                    for key, pattern in patterns.items():
                        match = re.search(pattern, final_section)
                        if match:
                            value = float(match.group(1))
                            metrics[key] = value

        time_match = re.search(r'Complete!\s+Total time:\s+([\d.]+)s', content)
        if time_match:
            metrics['computation_time'] = float(time_match.group(1))

    for key in metrics:
        if metrics[key] is not None:
            if isinstance(metrics[key], (np.integer, np.floating)):
                metrics[key] = float(metrics[key])
            elif isinstance(metrics[key], np.bool_):
                metrics[key] = bool(metrics[key])

    if metrics['avg_waiting_time'] is not None and metrics['objective_value'] is not None:
        return metrics
    return None


def load_scenario_results(archive_dir: str = "experiment_results") -> dict:
    """
    Load all scenario results from the archive directory.
    Only loads scenarios that have actual results.
    """
    print(f"{'~' * 5} LOADING SCENARIO RESULTS {'~' * 5}")

    results_dict = {}

    if not os.path.exists(archive_dir):
        print(f"[!] Archive directory '{archive_dir}' not found.")
        return results_dict

    scenario_dirs = [d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d))]

    for scenario_id in scenario_dirs:
        scenario_path = os.path.join(archive_dir, scenario_id)
        results_dict[scenario_id] = {}

        has_results = False

        for method in ['pso', 'aco']:
            method_path = os.path.join(scenario_path, method)
            if os.path.exists(method_path):
                log_files = [f for f in os.listdir(method_path) if f.startswith('run_') and f.endswith('.log')]

                if log_files:
                    log_file = sorted(log_files)[-1]
                    log_path = os.path.join(method_path, log_file)

                    metrics = parse_log_file(log_path)
                    if metrics:
                        results_dict[scenario_id][method.upper()] = metrics
                        has_results = True
                    else:
                        print(f"  [!] Could not parse results from {log_file}")
                else:
                    print(f"  [!] No log files found for {scenario_id}/{method}")
            else:
                print(f"  [!] No results directory for {scenario_id}/{method}")

        if not has_results:
            del results_dict[scenario_id]

    print(f"[*] Loaded results for {len(results_dict)} scenarios with actual data")
    print("~" * 60)

    return results_dict


def get_scenario_names() -> dict:
    """
    Get descriptive names for scenarios based on run_scenarios.py definitions.
    """
    return {
        '1A_low_traffic': 'Light Traffic (rho=0.33)',
        '1B_medium_traffic': 'Medium Traffic (rho=0.60)',
        '1C_high_traffic': 'High Traffic (rho=0.80)',
        '1D_near_saturation': 'Near Saturation (rho=0.95)',
        '1E_overloaded': 'Overloaded (rho=1.07)',
        '2A_MM1_low': 'M/M/1 Moderate Traffic',
        '2B_MG1_low': 'M/G/1 Moderate Traffic',
        '2C_MM1_high': 'M/M/1 High Traffic',
        '2D_MG1_high': 'M/G/1 High Traffic',
        '2E_MG1_high_regular': 'M/G/1 Regular High Traffic',
        '3A_symmetric_low': 'Symmetric Low Traffic',
        '3B_symmetric_high': 'Symmetric High Traffic',
        '3C_asymmetric_moderate': 'Asymmetric Moderate Traffic',
        '3D_asymmetric_rush_hour': 'Asymmetric Rush Hour',
        '4A_linear_3': 'Linear 3 Intersections',
        '4B_grid_2x2': '2x2 Grid',
        '4C_t_junction': 'T-Junction',
        '4D_grid_3x3': '3x3 Grid',
        '5A_algo_low_load': 'Algorithm Comparison Low Load',
        '5B_algo_high_load': 'Algorithm Comparison High Load',
        '5C_algo_near_saturation': 'Algorithm Comparison Near Saturation',
        '6A_tight_bounds': 'Tight Bounds',
        '6B_standard_bounds': 'Standard Bounds',
        '6C_wide_bounds': 'Wide Bounds',
        '7A_pso_w_low': 'PSO Low Inertia',
        '7B_pso_w_balanced': 'PSO Balanced Inertia',
        '7C_pso_w_high': 'PSO High Inertia',
        '7D_aco_q_low': 'ACO Low Locality',
        '7E_aco_q_balanced': 'ACO Balanced Locality',
        '7F_aco_q_high': 'ACO High Locality'
    }


def run_post_analysis(archive_dir: str = "experiment_results", output_dir: str = "results"):
    """
    Run comprehensive analysis on all completed scenarios.
    """
    print(f"{'~' * 5} POST-SCENARIO ANALYSIS {'~' * 5}")

    results_dict = load_scenario_results(archive_dir)

    if not results_dict:
        print("[!] No scenario results found. Run scenarios first with run_all_scenarios.bat")
        return

    scenario_names = get_scenario_names()

    pso_results = {}
    aco_results = {}

    for scenario_id, methods in results_dict.items():
        if 'PSO' in methods:
            pso_results[scenario_id] = methods['PSO']
        if 'ACO' in methods:
            aco_results[scenario_id] = methods['ACO']

    if not pso_results or not aco_results:
        print("[!] Need both PSO and ACO results for comparison")
        print(f"[*] PSO results: {len(pso_results)} scenarios")
        print(f"[*] ACO results: {len(aco_results)} scenarios")
        return

    print(f"[*] PSO results: {len(pso_results)} scenarios")
    print(f"[*] ACO results: {len(aco_results)} scenarios")

    print(f"\n{'~' * 5} RESULTS SUMMARY {'~' * 5}")
    for scenario_id in sorted(pso_results.keys()):
        pso = pso_results[scenario_id]
        aco = aco_results.get(scenario_id, {})

        name = scenario_names.get(scenario_id, scenario_id)
        print(f"\n{name}:")
        print(f"  PSO:  {pso['avg_waiting_time']:.2f}s wait, {pso['objective_value']:.2f} objective")
        if aco:
            print(f"  ACO:  {aco['avg_waiting_time']:.2f}s wait, {aco['objective_value']:.2f} objective")
            improvement = ((pso['avg_waiting_time'] - aco['avg_waiting_time']) / pso['avg_waiting_time'] * 100)
            print(f"  Improvement: {improvement:.1f}%")
    print("~" * 60)

    suite = IntegratedAnalysisSuite(output_dir=output_dir)

    analysis_results = suite.analyze_optimization_results(
        pso_results, aco_results, scenario_names
    )

    suite.save_publication_summary(analysis_results)

    print(f"\n{'~' * 5} ANALYSIS COMPLETE {'~' * 5}")
    print(f"[*] Results saved to: {output_dir}/")
    print("~" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-scenario analysis")
    parser.add_argument('--archive-dir', default='experiment_results',
                        help='Directory containing archived results')
    parser.add_argument('--output-dir', default='results',
                        help='Directory to save analysis outputs')

    args = parser.parse_args()

    run_post_analysis(args.archive_dir, args.output_dir)