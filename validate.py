"""
validate.py
===========
Model validation: compare simulated M/G/1 results against
Pollaczek-Khinchine (P-K) theoretical values across multiple
utilization levels (ρ).

Run once before dissertation submission:
    python validate.py

Produces:
    results/validation_table.txt   -- printable table for dissertation
    results/validation_plot.png    -- W_sim vs W_theory curve
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# patch config before importing project modules
import config
config.USE_ASYMMETRIC_TRAFFIC = False   # force uniform rates for clean theory comparison
config.ERLANG_K = 2                      # M/G/1 with Erlang-2
config.QUEUEING_MODEL = 'M/G/1'
config.NUM_INTERSECTIONS = 1             # single intersection -- matches theory exactly
config.NETWORK_TOPOLOGY = {0: []}        # isolated intersection, no connections
config.SIMULATION_DURATION = 3600        # 1 hour -- more stable estimates
config.WARMUP_PERIOD = 300               # 5 min warmup
config.NUM_SIMULATION_RUNS = 10          # 10 runs to reduce variance
config.VERBOSE = 0
config.SAVE_PLOTS = False
config.SHOW_PLOTS_DURING_OPT = False
config.INITIAL_GREEN_TIME = 30

from network import Network
from simulation import run_multiple_simulations
from metrics import validate_queue_theory

SERVICE_RATE = 0.4   # mu fixed throughout

# utilization levels to test (rho = lambda/mu)
# avoid ρ > 0.92 -- simulation becomes very slow near saturation
RHO_VALUES = [0.25, 0.375, 0.50, 0.625, 0.70, 0.75, 0.80, 0.875, 0.90, 0.925]


def run_validation():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    results = []

    print("=" * 70)
    print("  M/G/1 MODEL VALIDATION  (Simulated vs Pollaczek-Khinchine Theory)")
    print(f"  Service rate mu = {SERVICE_RATE}, Erlang-k = {config.ERLANG_K}")
    print(f"  Simulation duration = {config.SIMULATION_DURATION}s × {config.NUM_SIMULATION_RUNS} runs")
    print("=" * 70)
    print(f"\n{'ρ':>6}  {'λ':>6}  {'W_theory':>10}  {'W_sim':>10}  {'Error %':>8}  {'Verdict':>8}")
    print("-" * 60)

    for rho in RHO_VALUES:
        arrival_rate = rho * SERVICE_RATE
        config.ARRIVAL_RATE = arrival_rate

        # single-intersection network
        network = Network(
            num_intersections=1,
            topology={0: []},
            arrival_rate=arrival_rate,
            service_rate=SERVICE_RATE,
            initial_green=config.INITIAL_GREEN_TIME
        )

        # run simulation
        metrics = run_multiple_simulations(
            network=network,
            num_runs=config.NUM_SIMULATION_RUNS,
            duration=config.SIMULATION_DURATION,
            warmup=config.WARMUP_PERIOD,
            random_seed=42,
            verbose=0
        )

        # compare against theory
        validation = validate_queue_theory(
            lane_metrics=metrics,
            arrival_rate=arrival_rate,
            service_rate=SERVICE_RATE
        )

        if not validation['valid']:
            print(f"  rho={rho:.3f}  SKIPPED: {validation['reason']}")
            continue

        W_theory = validation['theoretical']['avg_waiting_time']
        W_sim    = validation['simulated']['avg_waiting_time']
        error    = validation['error_percent']['waiting_time']
        verdict  = "PASS" if error < 10.0 else "WARN" if error < 20.0 else "FAIL"

        print(f"{rho:>6.3f}  {arrival_rate:>6.3f}  {W_theory:>10.3f}  {W_sim:>10.3f}  {error:>7.2f}%  {verdict:>8}")

        results.append({
            'rho': rho,
            'arrival_rate': arrival_rate,
            'W_theory': W_theory,
            'W_sim': W_sim,
            'std_W': metrics.get('std_waiting_time', 0),
            'error_pct': error,
            'verdict': verdict
        })

    print("-" * 60)
    if results:
        avg_error = np.mean([r['error_pct'] for r in results])
        max_error = np.max([r['error_pct'] for r in results])
        passed    = sum(1 for r in results if r['verdict'] == 'PASS')
        print(f"\n  Average error: {avg_error:.2f}%   Max error: {max_error:.2f}%")
        print(f"  Passed (<10% error): {passed}/{len(results)}")

    print("\n")

    # save table to file
    _save_table(results)

    # save plot
    _save_plot(results)

    return results


def _save_table(results):
    path = os.path.join(config.OUTPUT_DIR, 'validation_table.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("M/G/1 Model Validation — Simulated vs Pollaczek-Khinchine Theory\n")
        f.write(f"Service rate mu={SERVICE_RATE}, Erlang-k={config.ERLANG_K}, "
                f"Simulation={config.SIMULATION_DURATION}s × {config.NUM_SIMULATION_RUNS} runs\n\n")
        f.write(f"{'rho':>6}  {'lambda':>7}  {'W_theory':>10}  {'W_sim':>10}  "
                f"{'Std_W':>7}  {'Error%':>8}  {'Verdict':>8}\n")
        f.write("-" * 65 + "\n")
        for r in results:
            f.write(f"{r['rho']:>6.3f}  {r['arrival_rate']:>7.4f}  "
                    f"{r['W_theory']:>10.3f}  {r['W_sim']:>10.3f}  "
                    f"{r['std_W']:>7.3f}  {r['error_pct']:>7.2f}%  {r['verdict']:>8}\n")
        if results:
            f.write("-" * 65 + "\n")
            f.write(f"  Average error: {np.mean([r['error_pct'] for r in results]):.2f}%\n")
            f.write(f"  Max error:     {np.max([r['error_pct'] for r in results]):.2f}%\n")
    print(f"  Table saved -> {path}")


def _save_plot(results):
    if not results:
        return

    rhos      = [r['rho'] for r in results]
    W_theory  = [r['W_theory'] for r in results]
    W_sim     = [r['W_sim'] for r in results]
    std_W     = [r['std_W'] for r in results]

    # also plot theoretical curve for full rho range
    rho_curve = np.linspace(0.05, 0.97, 200)
    mu = SERVICE_RATE
    k  = config.ERLANG_K
    # P-K formula for Erlang-k service
    second_moment = (k + 1) / (k * mu ** 2)
    W_curve = []
    for rho in rho_curve:
        lam = rho * mu
        Wq  = (lam * second_moment) / (2 * (1 - rho))
        W   = Wq + 1 / mu
        W_curve.append(W)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: W vs rho ---
    ax = axes[0]
    ax.plot(rho_curve, W_curve, 'b-', linewidth=2, label='P-K Theory (M/G/1, k=2)')
    ax.errorbar(rhos, W_sim, yerr=std_W, fmt='ro', markersize=7,
                capsize=4, label='Simulation (mean ± 1 std)')
    ax.set_xlabel('Utilization rho = lambda/mu', fontsize=12)
    ax.set_ylabel('Average Waiting Time W (seconds)', fontsize=12)
    ax.set_title('M/G/1 Validation: Simulated vs Theoretical W', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4, label='Instability (ρ=1)')

    # --- right: error % ---
    ax2 = axes[1]
    errors = [r['error_pct'] for r in results]
    colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in errors]
    bars = ax2.bar(rhos, errors, width=0.04, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=1.5, label='10% threshold (PASS)')
    ax2.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, label='20% threshold (WARN)')
    ax2.set_xlabel('Utilization ρ', fontsize=12)
    ax2.set_ylabel('Relative Error |W_theory - W_sim| / W_theory (%)', fontsize=11)
    ax2.set_title('Simulation Error vs Theory', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, 'validation_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved  -> {path}")


if __name__ == "__main__":
    run_validation()
    print("Validation complete.\n")