import numpy as np
from typing import Dict

import config
import os
from network import Network
from simulation import run_multiple_simulations
from metrics import calculate_objective_function, format_metrics_report, calculate_improvement
from optimization import PSO
from visualization import generate_all_plots, create_traffic_animation


class ObjectiveFunctionWrapper:
    """Picklable wrapper for PSO objective function evaluation."""
    
    def __init__(self, network: Network):
        self.network = network
    
    def __call__(self, timings: np.ndarray) -> float:
        """
        Evaluate fitness of given signal timings.
        Lower is better.
        """
        # update network with these timings
        self.network.update_signal_timings(timings)
        
        # run simulation
        metrics = run_multiple_simulations(
            network=self.network,
            num_runs=config.NUM_SIMULATION_RUNS,
            duration=config.SIMULATION_DURATION,
            warmup=config.WARMUP_PERIOD,
            random_seed=config.RANDOM_SEED,
            verbose=0  # silent during optimization
        )
        
        # calculate objective
        obj = calculate_objective_function(
            metrics,
            config.OBJECTIVE_WEIGHTS,
            config.MAX_QUEUE_THRESHOLD
        )
        
        return obj
    

def create_network() -> Network:
    """
    Create traffic network based on config parameters.
    """
    print(f"{'~' * 5} CREATING TRAFFIC NETWORK {'~' * 5}")
    
    network = Network(
        num_intersections=config.NUM_INTERSECTIONS,
        topology=config.NETWORK_TOPOLOGY,
        arrival_rate=config.ARRIVAL_RATE,
        service_rate=config.SERVICE_RATE,
        initial_green=config.INITIAL_GREEN_TIME
    )
    
    print(f"Network: {network}")
    print(f"Total lanes: {len(network.get_all_lanes())}")
    print(f"Arrival rate (λ): {config.ARRIVAL_RATE} vehicles/second")
    print(f"Service rate (μ): {config.SERVICE_RATE} vehicles/second")
    print(f"Utilization (ρ): {config.ARRIVAL_RATE/config.SERVICE_RATE:.2f}")
    
    # check stability
    if config.ARRIVAL_RATE >= config.SERVICE_RATE:
        print("\n[!]  WARNING: System is unstable (ρ >= 1)!")
        print("   Queues will grow indefinitely. Reduce arrival rate or increase service rate.")
    else:
        print(f"[*] System is stable (ρ < 1)")
    
    print("~" * 60 + "\n")
    
    return network


def run_baseline(network: Network) -> tuple:
    """
    Run baseline simulation with initial signal timings.
    """
    print(f"{'~' * 5} BASELINE SIMULATION {'~' * 5}")
    print(f"Initial green time: {config.INITIAL_GREEN_TIME}s (both directions)")
    print(f"Simulation duration: {config.SIMULATION_DURATION}s")
    print(f"Number of runs: {config.NUM_SIMULATION_RUNS}")
    print("~" * 60 + "\n")
    
    metrics = run_multiple_simulations(
        network=network,
        num_runs=config.NUM_SIMULATION_RUNS,
        duration=config.SIMULATION_DURATION,
        warmup=config.WARMUP_PERIOD,
        random_seed=config.RANDOM_SEED,
        verbose=config.VERBOSE
    )
    
    objective = calculate_objective_function(
        metrics, 
        config.OBJECTIVE_WEIGHTS, 
        config.MAX_QUEUE_THRESHOLD
    )
    
    print("\n" + format_metrics_report(metrics, objective))
    
    # store baseline timings
    baseline_timings = np.array([config.INITIAL_GREEN_TIME] * (config.NUM_INTERSECTIONS * 2))
    
    return metrics, objective, baseline_timings


def optimize_with_pso(network: Network) -> tuple:
    """
    Optimize signal timings using Particle Swarm Optimization.
    """
    # number of decision variables: 2 per intersection (NS green, EW green)
    num_variables = config.NUM_INTERSECTIONS * 2
    
    objective_function = ObjectiveFunctionWrapper(network)
    
    # create PSO optimizer
    pso = PSO(
        objective_function=objective_function,
        num_variables=num_variables,
        bounds=config.PSO_CONFIG['bounds'],
        config=config.PSO_CONFIG
    )
    
    # run optimization
    best_timings, best_score = pso.optimize(verbose=config.VERBOSE)
    
    # evaluate best solution with detailed metrics
    network.update_signal_timings(best_timings)
    best_metrics = run_multiple_simulations(
        network=network,
        num_runs=config.NUM_SIMULATION_RUNS,
        duration=config.SIMULATION_DURATION,
        warmup=config.WARMUP_PERIOD,
        random_seed=config.RANDOM_SEED,
        verbose=config.VERBOSE
    )
    
    print(f"\n{'~' * 5} PSO RESULTS {'~' * 5}")
    print(f"Optimized timings: {best_timings}")
    print(format_metrics_report(best_metrics, best_score))
    
    return best_metrics, best_score, best_timings, pso.get_history()


# TODO: add ACO

def compare_results(baseline_metrics: Dict, pso_metrics: Dict, aco_metrics: Dict = None):
    """
    Print comparison table of all methods.
    """
    print(f"{'~' * 5} FINAL COMPARISON {'~' * 5}")
    
    # calculate improvements
    pso_improvements = calculate_improvement(baseline_metrics, pso_metrics)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'PSO':<15} {'Improvement':<15}")
    print("-" * 75)
    
    print(f"{'Avg Waiting Time (s)':<30} "
          f"{baseline_metrics['avg_waiting_time']:<15.2f} "
          f"{pso_metrics['avg_waiting_time']:<15.2f} "
          f"{pso_improvements.get('avg_waiting_time', 0):<15.2f}%")
    
    print(f"{'Max Queue Length':<30} "
          f"{baseline_metrics['max_queue_length']:<15.1f} "
          f"{pso_metrics['max_queue_length']:<15.1f} "
          f"{pso_improvements.get('max_queue_length', 0):<15.2f}%")
    
    # TODO: ACO
    print("~" * 60 + "\n")


def main():
    """
    Main execution workflow.
    """
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    print(f"\n{'~' * 5} TRAFFIC SIGNAL OPTIMIZATION {'~' * 5}")
    print("M/M/1 Queueing Model + PSO Optimization")
    print(f"{'~' * 60}")

    network = create_network()
    
    # run baseline
    baseline_metrics, baseline_obj, baseline_timings = run_baseline(network)
    
    # optimize with PSO
    pso_metrics, pso_obj, pso_timings, pso_history = optimize_with_pso(network)

    # compare results
    compare_results(baseline_metrics, pso_metrics)

    if config.SAVE_PLOTS:
        generate_all_plots(
            baseline_metrics=baseline_metrics,
            pso_metrics=pso_metrics,
            pso_history=pso_history,
            baseline_timings=baseline_timings,
            optimized_timings=pso_timings,
            num_intersections=config.NUM_INTERSECTIONS,
            output_dir=config.OUTPUT_DIR
        )

        print("\nGenerating traffic animation...")
        create_traffic_animation(
            network=network,
            simulation_data=pso_metrics,  # has queue_samples and light_states
            save_path=os.path.join(config.OUTPUT_DIR, 'traffic_animation.mp4')
        )

        # keep plots visible
        import matplotlib.pyplot as plt
        plt.show(block=True)
    
    print(f"\n{'~' * 5} OPTIMIZATION COMPLETE {'~' * 5}")
    print(f"Results saved to '{config.OUTPUT_DIR}/' directory")
    print(f"{'~' * 60}\n")


if __name__ == "__main__":
    main()