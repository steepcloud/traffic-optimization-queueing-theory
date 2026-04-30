"""
Post-Scenario Analysis Integration

This script should be run after completing scenarios with run_all_scenarios.bat
to generate comprehensive analysis of all results.
"""

import os
import json
from integrated_analysis import IntegratedAnalysisSuite


def load_scenario_results(archive_dir: str = "experiment_results") -> dict:
    """
    Load all scenario results from the archive directory.
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

        for method in ['pso', 'aco']:
            method_path = os.path.join(scenario_path, method)
            if os.path.exists(method_path):
                results_file = os.path.join(method_path, 'optimization_results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results_dict[scenario_id][method.upper()] = json.load(f)
                else:
                    print(f"  [!] No results file found for {scenario_id}/{method}")

    print(f"[*] Loaded results for {len(results_dict)} scenarios")
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
        return

    print(f"[*] PSO results: {len(pso_results)} scenarios")
    print(f"[*] ACO results: {len(aco_results)} scenarios")

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