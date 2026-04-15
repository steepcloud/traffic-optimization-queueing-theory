"""
run_scenarios.py
================
Automated scenario runner for traffic signal optimization experiments.

For each scenario:
  1. Patches config.py with scenario-specific values
  2. Runs main.py --method pso  → copies results to results/<scenario_name>/pso/
  3. Runs main.py --method aco  → copies results to results/<scenario_name>/aco/
  4. Restores config.py to original

Usage:
    python run_scenarios.py                   # run all scenarios
    python run_scenarios.py --list            # print scenario names and exit
    python run_scenarios.py --only 1A 2C 5B  # run specific scenarios only
    python run_scenarios.py --skip 4D         # skip specific scenarios
    python run_scenarios.py --methods pso     # run only one method
"""

import os
import sys
import shutil
import subprocess
import argparse
import re
from datetime import datetime
from copy import deepcopy


# ---------------------------------------------------------------------------
# Scenario definitions
# Each scenario is a dict of config.py overrides.
# Keys must match variable names EXACTLY as they appear in config.py.
# ---------------------------------------------------------------------------

TOPOLOGY_2x2 = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

TOPOLOGY_LINEAR_3 = {
    0: [1],
    1: [0, 2],
    2: [1]
}

TOPOLOGY_T_3 = {
    0: [1],
    1: [0, 2],
    2: [1]
}

TOPOLOGY_3x3 = {
    0: [1, 3],    1: [0, 2, 4],    2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7],    7: [4, 6, 8],    8: [5, 7]
}

ASYMMETRIC_MODERATE = {
    0: {'N': 0.35, 'S': 0.38, 'E': 0.39, 'W': 0.36},
    1: {'N': 0.38, 'S': 0.35, 'E': 0.36, 'W': 0.39},
    2: {'N': 0.39, 'S': 0.36, 'E': 0.35, 'W': 0.38},
    3: {'N': 0.36, 'S': 0.39, 'E': 0.38, 'W': 0.35}
}

ASYMMETRIC_RUSH_HOUR = {
    0: {'N': 0.15, 'S': 0.38, 'E': 0.36, 'W': 0.15},
    1: {'N': 0.38, 'S': 0.15, 'E': 0.15, 'W': 0.36},
    2: {'N': 0.15, 'S': 0.36, 'E': 0.38, 'W': 0.15},
    3: {'N': 0.36, 'S': 0.15, 'E': 0.15, 'W': 0.38}
}

ASYMMETRIC_T_CENTER = {
    0: {'N': 0.20, 'S': 0.20, 'E': 0.20, 'W': 0.20},
    1: {'N': 0.35, 'S': 0.35, 'E': 0.35, 'W': 0.35},
    2: {'N': 0.20, 'S': 0.20, 'E': 0.20, 'W': 0.20}
}

SCENARIOS = {

    # -----------------------------------------------------------------------
    # GROUP 1: Traffic Load Scenarios
    # -----------------------------------------------------------------------
    "1A_low_traffic": {
        "_description": "Best Case - Low Traffic (rho=0.375)",
        "ARRIVAL_RATE": 0.15,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "1B_medium_traffic": {
        "_description": "Medium Case - Moderate Traffic (rho=0.625)",
        "ARRIVAL_RATE": 0.25,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "1C_high_traffic": {
        "_description": "High Traffic - Current Config (rho=0.875)",
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "1D_near_saturation": {
        "_description": "Worst Case - Near Saturation (rho=0.95)",
        "ARRIVAL_RATE": 0.38,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "1E_overloaded": {
        "_description": "Absolute Worst Case - Overloaded (rho=1.05, UNSTABLE)",
        "ARRIVAL_RATE": 0.42,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },

    # -----------------------------------------------------------------------
    # GROUP 2: Queueing Model Comparison (M/M/1 vs M/G/1)
    # -----------------------------------------------------------------------
    "2A_MM1_low": {
        "_description": "M/M/1 - Low Traffic (rho=0.625)",
        "ERLANG_K": 1,
        "ARRIVAL_RATE": 0.25,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "QUEUEING_MODEL": "M/M/1",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "2B_MG1_low": {
        "_description": "M/G/1 Erlang-2 - Low Traffic (rho=0.625)",
        "ERLANG_K": 2,
        "ARRIVAL_RATE": 0.25,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "QUEUEING_MODEL": "M/G/1",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "2C_MM1_high": {
        "_description": "M/M/1 - High Traffic (rho=0.875)",
        "ERLANG_K": 1,
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "QUEUEING_MODEL": "M/M/1",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "2D_MG1_high": {
        "_description": "M/G/1 Erlang-2 - High Traffic (rho=0.875)",
        "ERLANG_K": 2,
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "QUEUEING_MODEL": "M/G/1",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "2E_MG1_high_regular": {
        "_description": "M/G/1 Erlang-5 - High Traffic, Very Regular Arrivals (rho=0.875)",
        "ERLANG_K": 5,
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "QUEUEING_MODEL": "M/G/1",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },

    # -----------------------------------------------------------------------
    # GROUP 3: Asymmetric vs Symmetric Traffic
    # -----------------------------------------------------------------------
    "3A_symmetric_low": {
        "_description": "Symmetric Traffic - Low Load",
        "ARRIVAL_RATE": 0.25,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "3B_symmetric_high": {
        "_description": "Symmetric Traffic - High Load",
        "ARRIVAL_RATE": 0.35,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "3C_asymmetric_moderate": {
        "_description": "Asymmetric Traffic - Moderate (rho varies 0.875-0.975)",
        "USE_ASYMMETRIC_TRAFFIC": True,
        "LANE_ARRIVAL_RATES": ASYMMETRIC_MODERATE,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "ARRIVAL_RATE": 0.35,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "3D_asymmetric_rush_hour": {
        "_description": "Asymmetric Traffic - Extreme Rush Hour (directional)",
        "USE_ASYMMETRIC_TRAFFIC": True,
        "LANE_ARRIVAL_RATES": ASYMMETRIC_RUSH_HOUR,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "ARRIVAL_RATE": 0.35,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },

    # -----------------------------------------------------------------------
    # GROUP 4: Network Topology Scenarios
    # -----------------------------------------------------------------------
    "4A_linear_3": {
        "_description": "Linear Corridor - 3 Intersections",
        "NUM_INTERSECTIONS": 3,
        "NETWORK_TOPOLOGY": TOPOLOGY_LINEAR_3,
        "ARRIVAL_RATE": 0.25,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "4B_grid_2x2": {
        "_description": "2x2 Grid - 4 Intersections (current)",
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "ERLANG_K": 2,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "4C_t_junction": {
        "_description": "T-Junction - 3 Intersections, Center Busiest",
        "NUM_INTERSECTIONS": 3,
        "NETWORK_TOPOLOGY": TOPOLOGY_T_3,
        "USE_ASYMMETRIC_TRAFFIC": True,
        "LANE_ARRIVAL_RATES": ASYMMETRIC_T_CENTER,
        "SERVICE_RATE": 0.4,
        "ARRIVAL_RATE": 0.25,
        "ERLANG_K": 2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "4D_grid_3x3": {
        "_description": "3x3 Grid - 9 Intersections (18 decision variables)",
        "NUM_INTERSECTIONS": 9,
        "NETWORK_TOPOLOGY": TOPOLOGY_3x3,
        "ARRIVAL_RATE": 0.25,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },

    # -----------------------------------------------------------------------
    # GROUP 5: PSO vs ACO Algorithm Comparison (both methods always run)
    # -----------------------------------------------------------------------
    "5A_algo_low_load": {
        "_description": "Algorithm Comparison - Low Load (rho=0.625)",
        "ARRIVAL_RATE": 0.25,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "5B_algo_high_load": {
        "_description": "Algorithm Comparison - High Load + Asymmetric (rho=0.875)",
        "ARRIVAL_RATE": 0.35,
        "USE_ASYMMETRIC_TRAFFIC": True,
        "LANE_ARRIVAL_RATES": ASYMMETRIC_MODERATE,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },
    "5C_algo_near_saturation": {
        "_description": "Algorithm Comparison - Near Saturation (rho=0.95)",
        "ARRIVAL_RATE": 0.38,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
    },

    # -----------------------------------------------------------------------
    # GROUP 6: Signal Timing Bounds
    # -----------------------------------------------------------------------
    "6A_tight_bounds": {
        "_description": "Tight Signal Bounds [15, 60]",
        "MIN_GREEN_TIME": 15,
        "MAX_GREEN_TIME": 60,
        "ARRIVAL_RATE": 0.35,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
    },
    "6B_standard_bounds": {
        "_description": "Standard Signal Bounds [20, 90] (current)",
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "ARRIVAL_RATE": 0.35,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
    },
    "6C_wide_bounds": {
        "_description": "Wide Signal Bounds [10, 120]",
        "MIN_GREEN_TIME": 10,
        "MAX_GREEN_TIME": 120,
        "ARRIVAL_RATE": 0.35,
        "ERLANG_K": 2,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
    },

    # -----------------------------------------------------------------------
    # GROUP 7: Hyperparameter Sensitivity
    # _methods locks each scenario to only run the relevant algorithm.
    # 7A/7B/7C -> PSO only (vary inertia w)
    # 7D/7E/7F -> ACO only (vary locality q)
    # -----------------------------------------------------------------------

    # PSO inertia weight sensitivity (w): exploitation vs exploration tradeoff
    "7A_pso_w_low": {
        "_description": "PSO Sensitivity: Low inertia w=0.4 (exploitation-heavy)",
        "_methods": ["pso"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "PSO_CONFIG__w": 0.4,
    },
    "7B_pso_w_balanced": {
        "_description": "PSO Sensitivity: Balanced inertia w=0.7 (default)",
        "_methods": ["pso"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "PSO_CONFIG__w": 0.7,
    },
    "7C_pso_w_high": {
        "_description": "PSO Sensitivity: High inertia w=0.9 (exploration-heavy)",
        "_methods": ["pso"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "PSO_CONFIG__w": 0.9,
    },

    # ACO locality parameter sensitivity (q): archive exploitation vs exploration
    "7D_aco_q_low": {
        "_description": "ACO Sensitivity: Low locality q=0.2 (exploit best solutions)",
        "_methods": ["aco"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "ACO_CONFIG__q": 0.2,
    },
    "7E_aco_q_balanced": {
        "_description": "ACO Sensitivity: Balanced locality q=0.5 (default)",
        "_methods": ["aco"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "ACO_CONFIG__q": 0.5,
    },
    "7F_aco_q_high": {
        "_description": "ACO Sensitivity: High locality q=0.8 (broad exploration)",
        "_methods": ["aco"],
        "ARRIVAL_RATE": 0.35,
        "SERVICE_RATE": 0.4,
        "USE_ASYMMETRIC_TRAFFIC": False,
        "ERLANG_K": 2,
        "NUM_INTERSECTIONS": 4,
        "NETWORK_TOPOLOGY": TOPOLOGY_2x2,
        "MIN_GREEN_TIME": 20,
        "MAX_GREEN_TIME": 90,
        "ACO_CONFIG__q": 0.8,
    },
}


# ---------------------------------------------------------------------------
# Config patching logic
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> str:
    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_config(config_path: str, content: str):
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)


def patch_config(original: str, overrides: dict) -> str:
    """
    Apply overrides to the config file content.
    Handles scalars, booleans, strings, dicts.
    For MIN_GREEN_TIME / MAX_GREEN_TIME also patches PSO_CONFIG and ACO_CONFIG bounds inline.
    """
    content = original

    for key, value in overrides.items():
        if key.startswith('_'):
            continue  # skip metadata keys like _description

        if isinstance(value, dict):
            # replace the entire dict assignment (multi-line)
            content = _replace_dict_var(content, key, value)
        elif isinstance(value, bool):
            content = _replace_scalar(content, key, str(value))
        elif isinstance(value, str):
            content = _replace_scalar(content, key, f"'{value}'")
        else:
            content = _replace_scalar(content, key, repr(value))
    
    # handle nested dict keys (e.g. "PSO_CONFIG__w": 0.4 patches PSO_CONFIG['w'])
    for key, value in overrides.items():
        if '__' not in key or key.startswith('_'):
            continue
        parent, child = key.split('__', 1)
        # use regex to find and replace the specific key inside the parent dict
        pattern = rf"('{re.escape(child)}'\s*:\s*)[\d.]+"
        replacement = rf"\g<1>{repr(value)}"
        # only replace within the parent dict block
        # find the parent dict in content first
        parent_match = re.search(rf'^{re.escape(parent)}\s*=\s*\{{', content, re.MULTILINE)
        if not parent_match:
            print(f"  [WARN] Parent key '{parent}' not found for nested override '{key}'")
            continue
        # find the end of the parent dict
        start = parent_match.start()
        brace_start = content.index('{', start)
        depth, i = 0, brace_start
        while i < len(content):
            if content[i] == '{': depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        # replace only within that block
        block = content[brace_start:end]
        new_block, n = re.subn(pattern, replacement, block)
        if n == 0:
            print(f"  [WARN] Nested key '{child}' not found inside '{parent}'")
        content = content[:brace_start] + new_block + content[end:]

    # After patching MIN/MAX_GREEN_TIME, also update the bounds tuples inside
    # PSO_CONFIG and ACO_CONFIG so they reflect the new values.
    min_g = overrides.get('MIN_GREEN_TIME')
    max_g = overrides.get('MAX_GREEN_TIME')
    if min_g is not None or max_g is not None:
        # extract current values from patched content if not both overridden
        if min_g is None:
            m = re.search(r'^MIN_GREEN_TIME\s*=\s*(\d+)', content, re.MULTILINE)
            min_g = int(m.group(1)) if m else 20
        if max_g is None:
            m = re.search(r'^MAX_GREEN_TIME\s*=\s*(\d+)', content, re.MULTILINE)
            max_g = int(m.group(1)) if m else 90
        bounds_str = f"({min_g}, {max_g})"
        # replace bounds in PSO_CONFIG and ACO_CONFIG
        content = re.sub(
            r"('bounds'\s*:\s*)\(MIN_GREEN_TIME,\s*MAX_GREEN_TIME\)",
            f"\\1{bounds_str}",
            content
        )

    return content


def _replace_scalar(content: str, key: str, value_repr: str) -> str:
    """Replace  KEY = <anything up to newline or inline comment>"""
    pattern = rf'^({re.escape(key)}\s*=\s*).*?([ \t]*(?:#[^\n]*)?)\s*$'
    replacement = rf'\g<1>{value_repr}\2'
    new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if n == 0:
        print(f"  [WARN] Key '{key}' not found in config.py — skipping.")
    return new_content


def _replace_dict_var(content: str, key: str, new_dict: dict) -> str:
    """
    Replace a multi-line dict assignment for `key` with a formatted version.
    Matches:  KEY = {  ...  }  (handles nested braces).
    """
    # find start of assignment
    start_pattern = re.compile(rf'^{re.escape(key)}\s*=\s*\{{', re.MULTILINE)
    m = start_pattern.search(content)
    if not m:
        print(f"  [WARN] Dict key '{key}' not found in config.py — skipping.")
        return content

    start = m.start()
    brace_start = content.index('{', start)
    depth = 0
    i = brace_start
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1

    new_repr = _format_dict(key, new_dict)
    return content[:start] + new_repr + content[end:]


def _format_dict(key: str, d: dict, indent: int = 0) -> str:
    """Format a dict as Python source with consistent indentation."""
    lines = [f"{key} = {{"]
    for k, v in d.items():
        k_repr = repr(k) if isinstance(k, str) else str(k)
        if isinstance(v, dict):
            # nested dict (e.g. LANE_ARRIVAL_RATES inner dicts)
            inner = ', '.join(f"'{ik}': {iv}" for ik, iv in v.items())
            lines.append(f"    {k_repr}: {{{inner}}},")
        else:
            lines.append(f"    {k_repr}: {repr(v)},")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results management
# ---------------------------------------------------------------------------

def copy_results(src_dir: str, dest_dir: str):
    """Copy everything from src_dir into dest_dir."""
    if not os.path.exists(src_dir):
        print(f"  [WARN] Source results dir '{src_dir}' not found — nothing to copy.")
        return
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print(f"  Copied results → {dest_dir}")


def clear_results(results_dir: str):
    """Remove all files/folders inside results/ to start fresh."""
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scenario(scenario_name: str, scenario: dict, methods: list,
                 config_path: str, results_dir: str, archive_root: str,
                 python_exe: str):
    desc = scenario.get('_description', scenario_name)
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"  {desc}")
    print(f"{'='*70}")

    # load original config
    original_config = load_config(config_path)

    try:
        # patch config
        patched = patch_config(original_config, scenario)
        save_config(config_path, patched)
        print(f"  config.py patched OK")

        locked_methods = scenario.get('_methods', None)
        effective_methods = locked_methods if locked_methods else methods
        for method in effective_methods:
            print(f"\n  >>> Running --method {method.upper()} ...")
            result = subprocess.run(
                [python_exe, "main.py", "--method", method],
                capture_output=False,   # let output flow to terminal
            )

            if result.returncode != 0:
                print(f"  [ERROR] main.py exited with code {result.returncode} "
                      f"for scenario {scenario_name} / {method}")

            # archive results
            dest = os.path.join(archive_root, scenario_name, method)
            copy_results(results_dir, dest)

            # write a small metadata file so you know what settings produced these
            meta_path = os.path.join(dest, "scenario_info.txt")
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f"Scenario: {scenario_name}\n")
                f.write(f"Method:   {method.upper()}\n")
                f.write(f"Description: {desc}\n")
                f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Config overrides:\n")
                for k, v in scenario.items():
                    if not k.startswith('_'):
                        f.write(f"  {k} = {v!r}\n")

            # clear results/ for next run
            clear_results(results_dir)

    finally:
        # ALWAYS restore original config
        save_config(config_path, original_config)
        print(f"  config.py restored to original.")


def main():
    parser = argparse.ArgumentParser(description="Automated scenario runner")
    parser.add_argument('--list', action='store_true', help='List all scenarios and exit')
    parser.add_argument('--only', nargs='+', metavar='SCENARIO',
                        help='Run only these scenario names (e.g. 1A 2C)')
    parser.add_argument('--skip', nargs='+', metavar='SCENARIO',
                        help='Skip these scenario names')
    parser.add_argument('--methods', nargs='+', choices=['pso', 'aco'],
                        default=['pso', 'aco'],
                        help='Which methods to run (default: pso aco)')
    parser.add_argument('--config', default='config.py',
                        help='Path to config.py (default: config.py)')
    parser.add_argument('--results-dir', default='results',
                        help='Results output dir used by main.py (default: results)')
    parser.add_argument('--archive-dir', default='experiment_results',
                        help='Root dir for archived scenario results (default: experiment_results)')
    parser.add_argument('--python', default='python',
                        help='Python executable to use (default: python)')
    args = parser.parse_args()

    if args.list:
        print(f"\nAvailable scenarios ({len(SCENARIOS)} total):\n")
        for name, sc in SCENARIOS.items():
            print(f"  {name:<30} {sc.get('_description', '')}")
        print()
        return

    # determine which scenarios to run
    names = list(SCENARIOS.keys())
    if args.only:
        # support short names like "1A" matching "1A_low_traffic"
        names = [n for n in names if any(n.startswith(o) for o in args.only)]
    if args.skip:
        names = [n for n in names if not any(n.startswith(s) for s in args.skip)]

    if not names:
        print("[ERROR] No matching scenarios found. Use --list to see available scenarios.")
        sys.exit(1)

    config_path = os.path.abspath(args.config)
    results_dir = os.path.abspath(args.results_dir)
    archive_root = os.path.abspath(args.archive_dir)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(archive_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Traffic Signal Optimization — Automated Scenario Runner")
    print(f"  Scenarios to run: {len(names)}")
    print(f"  Methods: {args.methods}")
    print(f"  Archive root: {archive_root}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    failed = []

    for name in names:
        try:
            run_scenario(
                scenario_name=name,
                scenario=SCENARIOS[name],
                methods=args.methods,
                config_path=config_path,
                results_dir=results_dir,
                archive_root=archive_root,
                python_exe=args.python,
            )
        except Exception as e:
            print(f"\n[FATAL] Scenario {name} crashed: {e}")
            failed.append(name)
            # still try to restore config
            try:
                original = load_config(config_path)
                save_config(config_path, original)
            except Exception:
                pass

    print(f"\n{'='*70}")
    print(f"  ALL SCENARIOS COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results archived in: {archive_root}/")
    if failed:
        print(f"  FAILED scenarios: {failed}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()