import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from typing import Dict, List
import os


def plot_optimization_convergence(pso_history: Dict, aco_history: Dict = None, 
                                   save_path: str = None):
    """
    Plot convergence curves for PSO and/or ACO.
    Shows how objective function improves over iterations.
    Works with either or both histories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- LEFT PLOT: individual convergence (PSO or ACO) ---
    if pso_history is not None:
        ax1.plot(pso_history['iterations'], pso_history['best_scores'],
                'b-', linewidth=2, label='PSO Best Score')
        ax1.plot(pso_history['iterations'], pso_history['avg_scores'],
                'b--', alpha=0.6, label='PSO Average Score')
        ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
    
    if aco_history is not None:
        ax1.plot(aco_history['iterations'], aco_history['best_scores'],
                'g-', linewidth=2, label='ACO Best Score')
        ax1.plot(aco_history['iterations'], aco_history['avg_scores'],
                'g--', alpha=0.6, label='ACO Average Score')
        ax1.set_title('ACO Convergence', fontsize=14, fontweight='bold')

    if pso_history is not None and aco_history is not None:
        ax1.set_title('PSO vs ACO Convergence', fontsize=14, fontweight='bold')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective Function Value', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- RIGHT PLOT: best score progress comparison ---
    if pso_history is not None:
        ax2.plot(pso_history['iterations'], pso_history['best_scores'],
                'b-', linewidth=2, label='PSO Best')

    if aco_history is not None:
        ax2.plot(aco_history['iterations'], aco_history['best_scores'],
                'g-', linewidth=2, label='ACO Best')

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Objective Function Value', fontsize=12)
    ax2.set_title('Best Solution Progress', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    plt.close()


def plot_comparison_bars(baseline_metrics: Dict, pso_metrics: Dict = None, 
                        aco_metrics: Dict = None, save_path: str = None):
    """
    Bar chart comparing baseline vs optimized performance.
    """
    metrics_to_compare = ['avg_waiting_time', 'max_queue_length']
    metric_labels = ['Avg Waiting Time (s)', 'Max Queue Length']
    
    x = np.arange(len(metrics_to_compare))

    num_methods = 1 + (pso_metrics is not None) + (aco_metrics is not None)
    width = 0.8 / num_methods
    offset = -0.4 + width / 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_values = [baseline_metrics[m] for m in metrics_to_compare]
    bars = []
    
    bars.append(ax.bar(
        x + offset, baseline_values, width,
        label='Baseline', color='#e74c3c', alpha=0.8
    ))
    offset += width
    
    if pso_metrics is not None:
        pso_values = [pso_metrics[m] for m in metrics_to_compare]
        bars.append(ax.bar(
            x + offset, pso_values, width,
            label='PSO Optimized', color='#3498db', alpha=0.8
        ))
        offset += width

    if aco_metrics is not None:
        aco_values = [aco_metrics[m] for m in metrics_to_compare]
        bars.append(ax.bar(
            x + offset, aco_values, width,
            label='ACO Optimized', color='#2ecc71', alpha=0.8
        ))
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Optimized', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.close()


def plot_improvement_percentages(baseline_metrics: Dict, pso_metrics: Dict = None, 
                                 aco_metrics: Dict = None, save_path: str = None):
    """
    Show percentage improvement from baseline.
    """
    metrics = ['avg_waiting_time', 'max_queue_length']
    metric_labels = ['Waiting Time', 'Queue Length']
    
    x = np.arange(len(metrics))
    
    method_count = (pso_metrics is not None) + (aco_metrics is not None)
    width = 0.8 / max(method_count, 1)
    offset = -0.4 + width / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []
    
    if pso_metrics is not None:
        pso_improvements = [
            (baseline_metrics[m] - pso_metrics[m]) / baseline_metrics[m] * 100
            for m in metrics
        ]
        bars.append(ax.bar(
            x + offset, pso_improvements, width,
            label='PSO', color='#3498db', alpha=0.8
        ))
        offset += width
    
    if aco_metrics is not None:
        aco_improvements = [
            (baseline_metrics[m] - aco_metrics[m]) / baseline_metrics[m] * 100
            for m in metrics
        ]
        bars.append(ax.bar(
            x + offset, aco_improvements, width,
            label='ACO', color='#2ecc71', alpha=0.8
        ))
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Improvement over Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=10,
                fontweight='bold'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement plot to {save_path}")

    plt.close()


def plot_signal_timings(baseline_timings: np.ndarray, optimized_timings: np.ndarray,
                       num_intersections: int, save_path: str = None):
    """
    Compare baseline vs optimized signal timings for each intersection.
    """
    rows = math.ceil(math.sqrt(num_intersections))
    cols = math.ceil(num_intersections / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = np.array(axes).flatten()
    
    for i in range(num_intersections):
        ax = axes[i]
        
        # extract timings for this intersection
        baseline_ns = baseline_timings[i * 2]
        baseline_ew = baseline_timings[i * 2 + 1]
        optimized_ns = optimized_timings[i * 2]
        optimized_ew = optimized_timings[i * 2 + 1]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width / 2, [baseline_ns, baseline_ew], width, 
                      label='Baseline', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width / 2, [optimized_ns, optimized_ew], width, 
                      label='Optimized', color='#3498db', alpha=0.8)
        
        ax.set_ylabel('Green Time (seconds)', fontsize=10)
        ax.set_title(f'Intersection {i}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['North-South', 'East-West'])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # hide unused subplot slots
    for j in range(num_intersections, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Traffic Signal Timing Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved signal timing plot to {save_path}")

    plt.close()

def plot_queue_evolution(queue_data: Dict[int, List], save_path: str = None):
    """
    Plot queue length evolution over time for each lane.

    Args:
        queue_data: Dictionary {lane_id: [(time, queue_length), ...]}
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(queue_data)))

    for idx, (lane_id, data) in enumerate(queue_data.items()):
        if len(data) > 0:
            times, queues = zip(*data)
            ax.plot(times, queues, label=f'Lane {lane_id}',
                   color=colors[idx], linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax.set_title('Queue Length Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved queue evolution plot to {save_path}")

    plt.close()


def plot_queue_distribution_histogram(baseline_queue_data: Dict[int, List],
                                      optimized_queue_data: Dict[int, List],
                                      save_path: str = None):
    """
    Plot histogram comparing queue length distributions between baseline and optimized.

    Args:
        baseline_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for baseline
        optimized_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for optimized
    """
    baseline_queues = []
    optimized_queues = []

    for lane_id, data in baseline_queue_data.items():
        if len(data) > 0:
            baseline_queues.extend([q for _, q in data])

    for lane_id, data in optimized_queue_data.items():
        if len(data) > 0:
            optimized_queues.extend([q for _, q in data])

    if len(baseline_queues) == 0 or len(optimized_queues) == 0:
        print("[WARN] No queue data available for distribution histogram")
        return

    # bins for histogram (0-5, 5-10, 10-15, 15-20, 20+)
    bins = [0, 5, 10, 15, 20, 100]  # last bin catches everything >20
    bin_labels = ['0-5', '5-10', '10-15', '15-20', '20+']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    baseline_counts, _ = np.histogram(baseline_queues, bins=bins)
    baseline_percentages = baseline_counts / len(baseline_queues) * 100

    bars1 = ax1.bar(bin_labels, baseline_percentages, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Queue Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    optimized_counts, _ = np.histogram(optimized_queues, bins=bins)
    optimized_percentages = optimized_counts / len(optimized_queues) * 100

    bars2 = ax2.bar(bin_labels, optimized_percentages, color='#3498db', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Optimized Queue Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    baseline_mean = np.mean(baseline_queues)
    optimized_mean = np.mean(optimized_queues)
    improvement = (baseline_mean - optimized_mean) / baseline_mean * 100

    fig.suptitle(f'Queue Length Distribution Comparison | '
                 f'Baseline Mean: {baseline_mean:.1f} vs Optimized Mean: {optimized_mean:.1f} '
                 f'({improvement:.1f}% improvement)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved queue distribution histogram to {save_path}")

    plt.close()


def plot_blockage_frequency(baseline_queue_data: Dict[int, List],
                           optimized_queue_data: Dict[int, List],
                           max_queue_threshold: float,
                           save_path: str = None):
    """
    Plot blockage frequency analysis showing how often lanes exceed threshold.

    Args:
        baseline_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for baseline
        optimized_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for optimized
        max_queue_threshold: Queue length threshold for blockage
    """
    baseline_blockages = {}
    optimized_blockages = {}

    for lane_id, data in baseline_queue_data.items():
        if len(data) > 0:
            blocked_count = sum(1 for _, q in data if q >= max_queue_threshold)
            baseline_blockages[lane_id] = (blocked_count / len(data)) * 100

    for lane_id, data in optimized_queue_data.items():
        if len(data) > 0:
            blocked_count = sum(1 for _, q in data if q >= max_queue_threshold)
            optimized_blockages[lane_id] = (blocked_count / len(data)) * 100

    if len(baseline_blockages) == 0:
        print("[WARN] No queue data available for blockage frequency analysis")
        return

    lane_ids = sorted(set(list(baseline_blockages.keys()) + list(optimized_blockages.keys())))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(lane_ids))
    width = 0.35

    baseline_values = [baseline_blockages.get(lane_id, 0) for lane_id in lane_ids]
    optimized_values = [optimized_blockages.get(lane_id, 0) for lane_id in lane_ids]

    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, optimized_values, width, label='Optimized',
                   color='#3498db', alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Lane ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Blockage Frequency (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Blockage Frequency by Lane (Threshold: {max_queue_threshold} vehicles)',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Lane {lane_id}' for lane_id in lane_ids], rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # heatmap showing blockage reduction
    # matrix: rows = lanes, cols = [baseline, optimized]
    blockage_matrix = np.array([
        [baseline_blockages.get(lane_id, 0) for lane_id in lane_ids],
        [optimized_blockages.get(lane_id, 0) for lane_id in lane_ids]
    ]).T  # transpose so lanes are rows

    im = ax2.imshow(blockage_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Baseline', 'Optimized'])
    ax2.set_yticks(np.arange(len(lane_ids)))
    ax2.set_yticklabels([f'Lane {lane_id}' for lane_id in lane_ids])

    for i in range(len(lane_ids)):
        for j in range(2):
            value = blockage_matrix[i, j]
            text_color = 'white' if value > 50 else 'black'
            ax2.text(j, i, f'{value:.1f}%', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    ax2.set_title('Blockage Frequency Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Lane ID', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Blockage Frequency (%)', fontsize=10)

    avg_baseline_blockage = np.mean(baseline_values)
    avg_optimized_blockage = np.mean(optimized_values)
    total_reduction = (avg_baseline_blockage - avg_optimized_blockage) / avg_baseline_blockage * 100 if avg_baseline_blockage > 0 else 0

    fig.suptitle(f'Blockage Analysis | Avg Baseline: {avg_baseline_blockage:.1f}% vs '
                 f'Avg Optimized: {avg_optimized_blockage:.1f}% ({total_reduction:.1f}% reduction)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved blockage frequency analysis to {save_path}")

    plt.close()


def plot_intersection_utilization_heatmap(network, arrival_rate: float, service_rate: float,
                                        save_path: str = None):
    """
    Plot intersection utilization heatmap showing utilization ratios (ρ = λ/μ).

    Args:
        network: Network object containing intersection information
        arrival_rate: Base arrival rate (λ)
        service_rate: Service rate (μ)
    """
    import config

    num_intersections = config.NUM_INTERSECTIONS
    utilizations = []

    for int_id in range(num_intersections):
        intersection = network.intersections[int_id]

        total_arrival = 0
        for direction, lane in intersection.lanes.items():
            if config.USE_ASYMMETRIC_TRAFFIC and int_id in config.LANE_ARRIVAL_RATES:
                lane_rate = config.LANE_ARRIVAL_RATES[int_id].get(direction, arrival_rate)
            else:
                lane_rate = arrival_rate
            total_arrival += lane_rate

        # utilization ρ = λ / μ
        utilization = total_arrival / service_rate
        utilizations.append(utilization)

    grid_size = int(np.ceil(np.sqrt(num_intersections)))
    utilization_grid = np.zeros((grid_size, grid_size))

    for i, util in enumerate(utilizations):
        row = i // grid_size
        col = i % grid_size
        utilization_grid[row, col] = util

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(utilization_grid, cmap='RdYlGn_r', vmin=0, vmax=1.2,
                   aspect='equal', interpolation='nearest')

    for i in range(grid_size):
        for j in range(grid_size):
            intersection_idx = i * grid_size + j
            if intersection_idx < num_intersections:
                util = utilization_grid[i, j]
                text_color = 'white' if util > 0.6 else 'black'
                ax.text(j, i, f'Int {intersection_idx}\nρ={util:.2f}',
                       ha='center', va='center', color=text_color,
                       fontsize=11, fontweight='bold')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(grid_size)])

    ax.set_title(f'Intersection Utilization Heatmap (ρ = λ/μ)\n'
                 f'Arrival Rate: {arrival_rate:.3f} veh/s | Service Rate: {service_rate:.3f} veh/s',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Network Column', fontsize=12, fontweight='bold')
    ax.set_ylabel('Network Row', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utilization Ratio (ρ)', fontsize=11, fontweight='bold')

    cbar.ax.axhline(y=1.0, color='black', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, 1.0, 'Unstable', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    cbar.ax.axhline(y=0.8, color='black', linewidth=1, linestyle='--')
    cbar.ax.text(1.5, 0.8, 'Near Sat', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))

    cbar.ax.axhline(y=0.6, color='black', linewidth=1, linestyle='--')
    cbar.ax.text(1.5, 0.6, 'Heavy', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    avg_utilization = np.mean(utilizations)
    max_utilization = np.max(utilizations)
    overloaded_count = sum(1 for u in utilizations if u > 1.0)

    stats_text = (f'Average ρ: {avg_utilization:.2f}\n'
                 f'Maximum ρ: {max_utilization:.2f}\n'
                 f'Overloaded Intersections: {overloaded_count}/{num_intersections}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved intersection utilization heatmap to {save_path}")

    plt.close()


def plot_time_of_day_performance(baseline_queue_data: Dict[int, List],
                                 optimized_queue_data: Dict[int, List],
                                 warmup_period: float,
                                 simulation_duration: float,
                                 save_path: str = None):
    """
    Plot time-of-day performance showing queue length evolution with warmup period marked.

    Args:
        baseline_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for baseline
        optimized_queue_data: Dictionary {lane_id: [(time, queue_length), ...]} for optimized
        warmup_period: Warmup period duration in seconds
        simulation_duration: Total simulation duration in seconds
    """
    def aggregate_queue_data(queue_data):
        time_points = []
        queue_lengths = []

        for lane_id, data in queue_data.items():
            if len(data) > 0:
                for time, queue in data:
                    time_points.append(time)
                    queue_lengths.append(queue)

        if len(time_points) == 0:
            return [], []

        sorted_indices = np.argsort(time_points)
        time_points = np.array(time_points)[sorted_indices]
        queue_lengths = np.array(queue_lengths)[sorted_indices]

        bucket_size = 30  # seconds
        max_time = int(np.max(time_points))
        time_buckets = np.arange(0, max_time + bucket_size, bucket_size)

        avg_queues = []
        for i in range(len(time_buckets) - 1):
            mask = (time_points >= time_buckets[i]) & (time_points < time_buckets[i + 1])
            if np.any(mask):
                avg_queues.append(np.mean(queue_lengths[mask]))
            else:
                avg_queues.append(0)

        return time_buckets[:-1], avg_queues

    baseline_times, baseline_queues = aggregate_queue_data(baseline_queue_data)
    optimized_times, optimized_queues = aggregate_queue_data(optimized_queue_data)

    if len(baseline_times) == 0 or len(optimized_times) == 0:
        print("[WARN] No queue data available for time-of-day performance plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(baseline_times, baseline_queues, 'r-', linewidth=2, label='Baseline',
           alpha=0.8, marker='o', markersize=3)
    ax.plot(optimized_times, optimized_queues, 'b-', linewidth=2, label='Optimized',
           alpha=0.8, marker='s', markersize=3)

    ax.axvspan(0, warmup_period, alpha=0.2, color='yellow', label='Warmup Period')
    ax.axvline(warmup_period, color='orange', linestyle='--', linewidth=2)

    ax.text(warmup_period + (simulation_duration - warmup_period) / 2,
           ax.get_ylim()[1] * 0.9, 'Measurement Period',
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # calculate statistics for measurement period only
    def get_measurement_stats(times, queues):
        mask = times >= warmup_period
        if np.any(mask):
            measurement_queues = np.array(queues)[mask]
            return np.mean(measurement_queues), np.max(measurement_queues)
        return 0, 0

    baseline_avg, baseline_max = get_measurement_stats(baseline_times, baseline_queues)
    optimized_avg, optimized_max = get_measurement_stats(optimized_times, optimized_queues)

    stats_text = (f'Measurement Period Statistics:\n'
                 f'Baseline - Avg: {baseline_avg:.1f}, Max: {baseline_max:.1f}\n'
                 f'Optimized - Avg: {optimized_avg:.1f}, Max: {optimized_max:.1f}\n'
                 f'Improvement - Avg: {(baseline_avg - optimized_avg)/baseline_avg*100:.1f}%, '
                 f'Max: {(baseline_max - optimized_max)/baseline_max*100:.1f}%')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax.set_title(f'Time-of-Day Performance Analysis\n'
                 f'Warmup: {warmup_period}s | Total: {simulation_duration}s',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, simulation_duration)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-of-day performance plot to {save_path}")

    plt.close()


def plot_scenario_comparison_summary(scenario_results: Dict[str, Dict],
                                   save_path: str = None):
    """
    Plot scenario comparison summary using parallel coordinates.

    Args:
        scenario_results: Dictionary {scenario_name: {metric_name: value, ...}}
                        Each scenario dict should have:
                        - avg_waiting_time: float
                        - max_queue_length: float
                        - blockage_frequency: float
                        - improvement_percentage: float
                        - computation_time: float (optional)
    """
    if len(scenario_results) == 0:
        print("[WARN] No scenario results available for comparison summary")
        return

    metrics = ['avg_waiting_time', 'max_queue_length', 'blockage_frequency',
              'improvement_percentage', 'computation_time']
    metric_labels = ['Avg Wait Time (s)', 'Max Queue Length', 'Blockage Freq (%)',
                   'Improvement (%)', 'Comp Time (s)']

    valid_scenarios = {}
    for scenario_name, results in scenario_results.items():
        if all(metric in results for metric in metrics):
            valid_scenarios[scenario_name] = results

    if len(valid_scenarios) == 0:
        print("[WARN] No valid scenarios with complete metrics for comparison")
        return

    scenario_names = list(valid_scenarios.keys())
    num_scenarios = len(scenario_names)
    num_metrics = len(metrics)

    data_matrix = np.zeros((num_scenarios, num_metrics))
    for i, scenario_name in enumerate(scenario_names):
        for j, metric in enumerate(metrics):
            data_matrix[i, j] = valid_scenarios[scenario_name][metric]

    # normalize each metric to [0, 1] range for better visualization
    normalized_data = data_matrix.copy()
    for j in range(num_metrics):
        min_val = np.min(data_matrix[:, j])
        max_val = np.max(data_matrix[:, j])
        if max_val > min_val:
            normalized_data[:, j] = (data_matrix[:, j] - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(16, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, num_scenarios))

    for i in range(num_scenarios):
        ax.plot(range(num_metrics), normalized_data[i, :],
               color=colors[i], linewidth=2, marker='o', markersize=8,
               label=scenario_names[i], alpha=0.8)

    ax.set_xticks(range(num_metrics))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax.set_title('Scenario Comparison Summary (Parallel Coordinates)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)

    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)

    for i in range(num_scenarios):
        for j in range(num_metrics):
            original_value = data_matrix[i, j]
            normalized_value = normalized_data[i, j]
            # only show values for some points to avoid clutter
            if i % 2 == 0 or j == 2:  # show for alternating scenarios and middle metric
                ax.text(j, normalized_value + 0.02, f'{original_value:.1f}',
                       ha='center', va='bottom', fontsize=7, color=colors[i], fontweight='bold')

    avg_improvement = np.mean(data_matrix[:, 3])  # improvement_percentage is at index 3
    avg_wait_time = np.mean(data_matrix[:, 0])  # avg_waiting_time is at index 0
    avg_blockage = np.mean(data_matrix[:, 2])  # blockage_frequency is at index 2

    stats_text = (f'Overall Statistics:\n'
                 f'Avg Improvement: {avg_improvement:.1f}%\n'
                 f'Avg Wait Time: {avg_wait_time:.1f}s\n'
                 f'Avg Blockage Freq: {avg_blockage:.1f}%\n'
                 f'Total Scenarios: {num_scenarios}')

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scenario comparison summary to {save_path}")

    plt.close()


def plot_algorithm_efficiency_scatter(pso_results: Dict[str, Dict],
                                    aco_results: Dict[str, Dict],
                                    save_path: str = None):
    """
    Plot algorithm efficiency scatter plot showing computation time vs improvement percentage.

    Args:
        pso_results: Dictionary {scenario_name: {'computation_time': float, 'improvement_percentage': float, ...}}
        aco_results: Dictionary {scenario_name: {'computation_time': float, 'improvement_percentage': float, ...}}
    """
    pso_scenarios = []
    pso_times = []
    pso_improvements = []

    aco_scenarios = []
    aco_times = []
    aco_improvements = []

    for scenario_name, results in pso_results.items():
        if 'computation_time' in results and 'improvement_percentage' in results:
            pso_scenarios.append(scenario_name)
            pso_times.append(results['computation_time'])
            pso_improvements.append(results['improvement_percentage'])

    for scenario_name, results in aco_results.items():
        if 'computation_time' in results and 'improvement_percentage' in results:
            aco_scenarios.append(scenario_name)
            aco_times.append(results['computation_time'])
            aco_improvements.append(results['improvement_percentage'])

    if len(pso_scenarios) == 0 and len(aco_scenarios) == 0:
        print("[WARN] No algorithm efficiency data available")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if len(pso_scenarios) > 0:
        ax.scatter(pso_times, pso_improvements, c='#3498db', s=150, alpha=0.7,
                  edgecolors='black', linewidth=2, label='PSO', marker='o', zorder=5)

        for i, scenario in enumerate(pso_scenarios):
            ax.annotate(scenario, (pso_times[i], pso_improvements[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8,
                       color='blue', fontweight='bold', alpha=0.8)

    if len(aco_scenarios) > 0:
        ax.scatter(aco_times, aco_improvements, c='#2ecc71', s=150, alpha=0.7,
                  edgecolors='black', linewidth=2, label='ACO', marker='s', zorder=5)

        for i, scenario in enumerate(aco_scenarios):
            ax.annotate(scenario, (aco_times[i], aco_improvements[i]),
                       textcoords="offset points", xytext=(5, -15), fontsize=8,
                       color='green', fontweight='bold', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    all_times = pso_times + aco_times
    all_improvements = pso_improvements + aco_improvements

    if len(all_times) > 0 and len(all_improvements) > 0:
        median_time = np.median(all_times)
        median_improvement = np.median(all_improvements)

        ax.axvline(x=median_time, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=median_improvement, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.text(median_time * 0.5, median_improvement * 1.5, 'Fast & Effective',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        ax.text(median_time * 1.5, median_improvement * 1.5, 'Slow but Effective',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.text(median_time * 0.5, median_improvement * 0.5, 'Fast but Ineffective',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

        ax.text(median_time * 1.5, median_improvement * 0.5, 'Slow & Ineffective',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))

    ax.set_xlabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Efficiency: Computation Time vs Improvement',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    stats_text = ""
    if len(pso_times) > 0:
        avg_pso_time = np.mean(pso_times)
        avg_pso_improvement = np.mean(pso_improvements)
        stats_text += f'PSO - Avg Time: {avg_pso_time:.1f}s, Avg Improvement: {avg_pso_improvement:.1f}%\n'

    if len(aco_times) > 0:
        avg_aco_time = np.mean(aco_times)
        avg_aco_improvement = np.mean(aco_improvements)
        stats_text += f'ACO - Avg Time: {avg_aco_time:.1f}s, Avg Improvement: {avg_aco_improvement:.1f}%'

    if stats_text:
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved algorithm efficiency scatter plot to {save_path}")

    plt.close()


def generate_all_plots(baseline_metrics: Dict,
                       num_intersections: int,
                       output_dir: str,
                       pso_metrics: Dict = None,
                       aco_metrics: Dict = None,
                       pso_history: Dict = None,
                       aco_history: Dict = None,
                       baseline_timings: np.ndarray = None,
                       optimized_timings: np.ndarray = None,
                       network = None,
                       arrival_rate: float = None,
                       service_rate: float = None,
                       warmup_period: float = None,
                       simulation_duration: float = None,
                       max_queue_threshold: float = None):
    """
    Generate and save all visualization plots.
    Works with PSO only, ACO only, or both.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"\n{'~' * 5} GENERATING VISUALIZATIONS {'~' * 5}")

    # 1. optimization convergence
    print("Generating convergence plot...")
    plot_optimization_convergence(
        pso_history=pso_history,
        aco_history=aco_history,
        save_path=os.path.join(output_dir, 'convergence.png')
    )

    # 2. comparison bars
    print("Generating comparison plot...")
    plot_comparison_bars(
        baseline_metrics=baseline_metrics,
        pso_metrics=pso_metrics,
        aco_metrics=aco_metrics,
        save_path=os.path.join(output_dir, 'comparison.png')
    )

    # 3. improvement percentages
    print("Generating improvement plot...")
    plot_improvement_percentages(
        baseline_metrics=baseline_metrics,
        pso_metrics=pso_metrics,
        aco_metrics=aco_metrics,
        save_path=os.path.join(output_dir, 'improvement.png')
    )

    # 4. signal timings
    if baseline_timings is not None and optimized_timings is not None:
        print("Generating signal timing plot...")
        plot_signal_timings(
            baseline_timings=baseline_timings,
            optimized_timings=optimized_timings,
            num_intersections=num_intersections,
            save_path=os.path.join(output_dir, 'signal_timings.png')
        )

    # 5. queue evolution (if data available)
    active_metrics = pso_metrics if pso_metrics is not None else aco_metrics
    if active_metrics is not None and 'queue_samples' in active_metrics:
        print("Generating queue evolution plot...")
        plot_queue_evolution(
            queue_data=active_metrics['queue_samples'],
            save_path=os.path.join(output_dir, 'queue_evolution.png')
        )

    # 6. queue distribution histogram (if both baseline and optimized data available)
    if (baseline_metrics is not None and 'queue_samples' in baseline_metrics and
        active_metrics is not None and 'queue_samples' in active_metrics):
        print("Generating queue distribution histogram...")
        plot_queue_distribution_histogram(
            baseline_queue_data=baseline_metrics['queue_samples'],
            optimized_queue_data=active_metrics['queue_samples'],
            save_path=os.path.join(output_dir, 'queue_distribution.png')
        )

    # 7. blockage frequency analysis (if threshold and queue data available)
    if (max_queue_threshold is not None and
        baseline_metrics is not None and 'queue_samples' in baseline_metrics and
        active_metrics is not None and 'queue_samples' in active_metrics):
        print("Generating blockage frequency analysis...")
        plot_blockage_frequency(
            baseline_queue_data=baseline_metrics['queue_samples'],
            optimized_queue_data=active_metrics['queue_samples'],
            max_queue_threshold=max_queue_threshold,
            save_path=os.path.join(output_dir, 'blockage_frequency.png')
        )

    # 8. intersection utilization heatmap (if network and rates available)
    if network is not None and arrival_rate is not None and service_rate is not None:
        print("Generating intersection utilization heatmap...")
        plot_intersection_utilization_heatmap(
            network=network,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            save_path=os.path.join(output_dir, 'utilization_heatmap.png')
        )

    # 9. time-of-day performance (if warmup and duration available)
    if (warmup_period is not None and simulation_duration is not None and
        baseline_metrics is not None and 'queue_samples' in baseline_metrics and
        active_metrics is not None and 'queue_samples' in active_metrics):
        print("Generating time-of-day performance plot...")
        plot_time_of_day_performance(
            baseline_queue_data=baseline_metrics['queue_samples'],
            optimized_queue_data=active_metrics['queue_samples'],
            warmup_period=warmup_period,
            simulation_duration=simulation_duration,
            save_path=os.path.join(output_dir, 'time_of_day_performance.png')
        )

    print(f"All plots saved to '{output_dir}/'")
    print("~" * 60 + "\n")


def plot_optimizer_live(optimizer, iteration: int, method: str, is_final: bool = False):
    """
    Unified real-time visualization for both PSO and ACO optimizers.
    Accepts method='pso' or method='aco'.

    Left plot:  solution scatter — always plots VALID (bounded) solutions:
                PSO -> personal best positions (always within bounds)
                ACO -> archive solutions       (always within bounds)
                Raw PSO particle positions are intentionally NOT plotted
                because inertia/velocity can push them outside bounds,
                causing the axes to collapse to a corner.

    Right plot: convergence curve — identical for both methods.

    Args:
        is_final: If True, this is the final iteration (including early stop)
    """
    import config

    _fn = plot_optimizer_live  # use function itself as state container

    # --- create figure on first call only ---
    if not hasattr(_fn, 'fig'):
        plt.ion()
        _fn.fig, _fn.axes = plt.subplots(1, 2, figsize=(15, 6))
        _fn.cbar = None
        plt.show(block=False)

    fig = _fn.fig
    ax1, ax2 = _fn.axes
    ax1.clear()
    ax2.clear()

    # Extract bounded solutions and scores depending on method
    if method == 'pso':
        # personal best positions are always clamped to bounds
        solutions_2d = optimizer.personal_best_positions[:, :2]
        scores       = list(optimizer.personal_best_scores)
        best_score   = optimizer.global_best_score
        best_sol_2d  = optimizer.global_best_position[:2]
        n_label      = f'Personal Bests (n={optimizer.num_particles})'
        best_label   = f'Global Best ({best_score:.2f})'
        title        = f'PSO Swarm - Iteration {iteration + 1}/{optimizer.num_iterations}'
        bounds       = optimizer.bounds
        weights      = None   # PSO has no weights
        ranks        = False

    else:  # aco
        archive_solutions = np.array([sol for _, sol in optimizer.archive])
        solutions_2d = archive_solutions[:, :2]
        scores       = [s for s, _ in optimizer.archive]
        best_score   = optimizer.global_best_score
        best_sol_2d  = solutions_2d[0]
        n_label      = f'Archive (k={optimizer.archive_size})'
        best_label   = f'Best (score={best_score:.2f})'
        title        = f'ACO Archive - Iteration {iteration + 1}/{optimizer.num_iterations}'
        bounds       = optimizer.bounds
        weights      = optimizer._compute_weights()
        ranks        = True

    # Left plot: solution scatter
    scatter = ax1.scatter(
        solutions_2d[:, 0], solutions_2d[:, 1],
        c=scores, cmap='RdYlGn_r', s=200, alpha=0.85,
        edgecolors='black', linewidth=1.5,
        vmin=min(scores), vmax=max(scores),
        label=n_label, zorder=5
    )

    # colorbar — create once, update after
    if _fn.cbar is None:
        _fn.cbar = plt.colorbar(scatter, ax=ax1)
        _fn.cbar.set_label('Fitness Score', fontsize=10)
    else:
        _fn.cbar.update_normal(scatter)

    # rank annotations (ACO only)
    if ranks:
        for rank, (sol, score) in enumerate(zip(solutions_2d, scores)):
            ax1.annotate(f'#{rank+1}', (sol[0], sol[1]),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=8, color='black', fontweight='bold')

    # weight circles (ACO only)
    if weights is not None:
        for sol, w in zip(solutions_2d, weights):
            ax1.add_patch(plt.Circle(
                (sol[0], sol[1]), radius=w * 15,
                fill=False, edgecolor='blue', linewidth=1.5,
                alpha=0.4, linestyle='--', zorder=3
            ))

    # global / archive best — big red star
    ax1.scatter(best_sol_2d[0], best_sol_2d[1],
               c='red', s=450, alpha=1.0, marker='*',
               edgecolors='black', linewidth=2,
               label=best_label, zorder=10)

    # fixed axes
    padding = 5
    ax1.set_xlim(bounds[0] - padding, bounds[1] + padding)
    ax1.set_ylim(bounds[0] - padding, bounds[1] + padding)

    # search bounds rectangle (dashed box)
    ax1.add_patch(plt.Rectangle(
        (bounds[0], bounds[0]),
        bounds[1] - bounds[0], bounds[1] - bounds[0],
        fill=False, edgecolor='gray', linewidth=2, linestyle='--', zorder=1
    ))

    ax1.set_xlabel('Intersection 0: NS Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Intersection 0: EW Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right plot: convergence curve - identical for both methods
    iters       = optimizer.history['iterations']
    best_scores = optimizer.history['best_scores']
    avg_scores  = optimizer.history['avg_scores']

    ax2.plot(iters, best_scores, 'b-', linewidth=2.5, label='Best Score',
             marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='white')
    ax2.plot(iters, avg_scores, 'orange', linewidth=2, linestyle='--', alpha=0.8,
             label='Average Score', marker='s', markersize=4)
    ax2.fill_between(iters, best_scores, avg_scores, alpha=0.2, color='blue')

    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Objective Function Value', fontsize=11, fontweight='bold')
    ax2.set_title('Convergence Progress', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    if len(best_scores) > 0:
        improvement = ((best_scores[0] - best_scores[-1]) / best_scores[0] * 100) if best_scores[0] > 0 else 0
        ax2.text(0.02, 0.98,
                f'Current Best: {best_score:.2f}\nImprovement: {improvement:.1f}%',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # save frame
    save_convergence_frame(fig, method, iteration, config.OUTPUT_DIR)

    # on final iteration (including early stop): save final PNG + stitch MP4, reset state for next run
    if is_final:
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        save_path = os.path.join(config.OUTPUT_DIR, f'{method}_optimizer_final.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Final {method.upper()} plot saved to {save_path}")
        stitch_convergence_animation(method, config.OUTPUT_DIR)
        # reset so next run (different scenario) gets a fresh figure
        del _fn.fig, _fn.axes, _fn.cbar


# --- backwards-compatible aliases so existing call sites need no changes ---
def plot_pso_particles_live(pso, iteration: int, is_final: bool = False):
    plot_optimizer_live(pso, iteration, method='pso', is_final=is_final)


def plot_aco_archive_live(aco, iteration: int, is_final: bool = False):
    plot_optimizer_live(aco, iteration, method='aco', is_final=is_final)


def save_convergence_frame(fig, method: str, iteration: int, output_dir: str):
    """Save current figure as a frame for convergence animation."""
    frames_dir = os.path.join(output_dir, f'{method}_convergence_frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_path = os.path.join(frames_dir, f'frame_{iteration:04d}.png')
    fig.savefig(frame_path, dpi=120, bbox_inches='tight')
    return frames_dir


def stitch_convergence_animation(method: str, output_dir: str):
    """
    Stitch all saved convergence frames into MP4.
    Called automatically after optimization finishes.
    """
    import glob

    frames_dir = os.path.join(output_dir, f'{method}_convergence_frames')
    frame_pattern = os.path.join(frames_dir, 'frame_*.png')
    frames = sorted(glob.glob(frame_pattern))

    if len(frames) == 0:
        print(f"[!] No frames found in {frames_dir}")
        return

    print(f"\nStitching {len(frames)} frames into convergence animation...")

    # load first frame to get dimensions
    import matplotlib.image as mpimg
    first_frame = mpimg.imread(frames[0])
    h, w = first_frame.shape[:2]

    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)

    im = ax.imshow(first_frame)

    def update_frame(i):
        img = mpimg.imread(frames[i])
        im.set_data(img)
        return [im]

    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=len(frames),
        interval=500,   # 500ms per frame = 2fps (slow enough to see each iteration)
        blit=True
    )

    save_path = os.path.join(output_dir, f'{method}_convergence.mp4')

    try:
        anim.save(save_path, writer='ffmpeg', fps=2, dpi=100)
        print(f"[SUCCESS] Convergence animation saved to {save_path}")

        # cleanup frames folder
        import shutil
        shutil.rmtree(frames_dir)
        print(f"[CLEANUP] Removed frames folder: {frames_dir}")

    except Exception as e:
        print(f"[FAILED] Could not stitch animation: {e}")
        print(f"  Individual frames saved in: {frames_dir}")

    plt.close(fig)


def create_traffic_animation(network, simulation_data: Dict, save_path: str = None):
    """
    Create professional animated visualization.
    """
    import config
    
    fig = plt.figure(figsize=(24, 16))
    
    # main plot
    ax = fig.add_subplot(111, position=[0.05, 0.05, 0.75, 0.85])  # [left, bottom, width, height]
    
    # get intersection positions
    base_positions = network.get_positions()
    
    positions = {}
    for int_id, (x, y) in base_positions.items():
        positions[int_id] = (x * 25.0, y * 25.0)

    # extract queue data
    queue_samples = simulation_data.get('queue_samples', {})
    
    max_time = 0
    for lane_id, samples in queue_samples.items():
        if len(samples) > 0:
            max_time = max(max_time, max([t for t, q in samples]))
    
    if max_time == 0:
        max_time = config.SIMULATION_DURATION
    
    time_step = 5
    num_frames = int(max_time / time_step)
    
    def get_queue_at_time(lane_id, time):
        if lane_id not in queue_samples or len(queue_samples[lane_id]) == 0:
            return 0
        samples = queue_samples[lane_id]
        for i, (t, q) in enumerate(samples):
            if t >= time:
                if i == 0:
                    return q
                t_prev, q_prev = samples[i - 1]
                ratio = (time - t_prev) / (t - t_prev) if t != t_prev else 0
                return q_prev + ratio * (q - q_prev)
        return samples[-1][1] if samples else 0
    
    def get_light_phase_at_time(int_id, time):
        if 'light_states' in simulation_data and int_id in simulation_data['light_states']:
            states = simulation_data['light_states'][int_id]
            for i, (t, phase) in enumerate(states):
                if t >= time:
                    return states[i-1][1] if i > 0 else phase
            return states[-1][1] if states else 0
        
        intersection = network.intersections[int_id]
        cycle_time = intersection.traffic_light.cycle_time
        time_in_cycle = time % cycle_time
        
        if time_in_cycle < intersection.traffic_light.green_ns:
            return 0
        elif time_in_cycle < intersection.traffic_light.green_ns + intersection.traffic_light.yellow:
            return 1
        elif time_in_cycle < intersection.traffic_light.green_ns + intersection.traffic_light.yellow + intersection.traffic_light.green_ew:
            return 2
        else:
            return 3
    
    def update(frame):
        fig.patches.clear()  # clear previous legend patches
        fig.texts.clear()    # clear previous legend texts
        ax.clear()
        current_time = frame * time_step
        
        # plot limits with padding
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        x_min, x_max = min(x_coords) - 20.0, max(x_coords) + 20.0
        y_min, y_max = min(y_coords) - 20.0, max(y_coords) + 20.0
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        
        # title with stats
        total_vehicles = 0
        max_queue_current = 0
        green_lights = 0
        red_lights = 0
        
        # draw roads
        for int_id, intersection in network.intersections.items():
            x1, y1 = positions[int_id]
            for direction, connected_id in intersection.outgoing_connections.items():
                if connected_id in positions:
                    x2, y2 = positions[connected_id]
                    ax.plot([x1, x2], [y1, y2], color='#7f8c8d', linewidth=40, 
                           alpha=0.3, solid_capstyle='round', zorder=1)
        
        # draw intersections
        for int_id, intersection in network.intersections.items():
            x, y = positions[int_id]
            phase = get_light_phase_at_time(int_id, current_time)
            
            # light colors
            ns_color = '#ff0000'
            ew_color = '#ff0000'
            
            if phase == 0:
                ns_color = '#00ff00'
                ew_color = '#ff0000'
            elif phase == 1:
                ns_color = '#ffff00'
                ew_color = '#ff0000'
            elif phase == 2:
                ns_color = '#ff0000'
                ew_color = '#00ff00'
            else:
                ns_color = '#ff0000'
                ew_color = '#ffff00'
            
            # intersection center
            center = plt.Circle((x, y), 0.8, color='#34495e', zorder=5)
            ax.add_patch(center)
            ax.text(x, y, str(int_id), color='white', ha='center', va='center',
                   fontweight='bold', fontsize=32, zorder=6)
            
            lane_offset = 10.0
            light_offset = 3.5
            bar_width = 0.6
            
            for direction, lane in intersection.lanes.items():
                queue_length = get_queue_at_time(lane.lane_id, current_time)
                total_vehicles += int(queue_length)
                max_queue_current = max(max_queue_current, queue_length)
                
                # bar length
                max_bar_length = 3.0
                bar_length = min(queue_length / 20.0, 1.0) * max_bar_length
                
                if queue_length < 5:
                    queue_color = '#2ecc71'
                elif queue_length < 10:
                    queue_color = '#f39c12'
                else:
                    queue_color = '#e74c3c'
                
                # queue bar starts right after the traffic light (light_offset + light_radius)
                queue_start_offset = light_offset + 0.6  # start right after light
                
                if direction == 'N':
                    # queue bar starts after the light
                    rect = plt.Rectangle((x - bar_width/2, y + queue_start_offset), bar_width, bar_length,
                                        facecolor=queue_color, alpha=0.85, zorder=3,
                                        edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                    
                    if queue_length > 0:
                        ax.text(x, y + queue_start_offset + bar_length + 0.6, f'{int(queue_length)}',
                               fontsize=16, ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                       edgecolor='black', linewidth=2, alpha=0.95))
                    
                    light = plt.Circle((x, y + light_offset), 0.5, facecolor=ns_color,
                                     edgecolor='black', linewidth=3, zorder=7)
                    ax.add_patch(light)
                    if ns_color == '#00ff00':
                        green_lights += 1
                    elif ns_color == '#ff0000':
                        red_lights += 1
                
                elif direction == 'S':
                    rect = plt.Rectangle((x - bar_width/2, y - queue_start_offset - bar_length), bar_width, bar_length,
                                        facecolor=queue_color, alpha=0.85, zorder=3,
                                        edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                    
                    if queue_length > 0:
                        ax.text(x, y - queue_start_offset - bar_length - 0.6, f'{int(queue_length)}',
                               fontsize=16, ha='center', va='top', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                       edgecolor='black', linewidth=2, alpha=0.95))
                    
                    light = plt.Circle((x, y - light_offset), 0.5, facecolor=ns_color,
                                     edgecolor='black', linewidth=3, zorder=7)
                    ax.add_patch(light)
                
                elif direction == 'E':
                    rect = plt.Rectangle((x + queue_start_offset, y - bar_width/2), bar_length, bar_width,
                                        facecolor=queue_color, alpha=0.85, zorder=3,
                                        edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                    
                    if queue_length > 0:
                        ax.text(x + queue_start_offset + bar_length + 0.6, y, f'{int(queue_length)}',
                               fontsize=16, ha='left', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                       edgecolor='black', linewidth=2, alpha=0.95))
                    
                    light = plt.Circle((x + light_offset, y), 0.5, facecolor=ew_color,
                                     edgecolor='black', linewidth=3, zorder=7)
                    ax.add_patch(light)
                    if ew_color == '#00ff00':
                        green_lights += 1
                    elif ew_color == '#ff0000':
                        red_lights += 1
                
                elif direction == 'W':
                    rect = plt.Rectangle((x - queue_start_offset - bar_length, y - bar_width/2), bar_length, bar_width,
                                        facecolor=queue_color, alpha=0.85, zorder=3,
                                        edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                    
                    if queue_length > 0:
                        ax.text(x - queue_start_offset - bar_length - 0.6, y, f'{int(queue_length)}',
                               fontsize=16, ha='right', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                       edgecolor='black', linewidth=2, alpha=0.95))
                    
                    light = plt.Circle((x - light_offset, y), 0.5, facecolor=ew_color,
                                     edgecolor='black', linewidth=3, zorder=7)
                    ax.add_patch(light)
        
        # legend outside plot area
        legend_x = 0.82
        
        # traffic Lights section
        y_base = 0.75
        fig.text(legend_x, y_base, '● Traffic Lights', fontsize=16, fontweight='bold', 
                transform=fig.transFigure, ha='left', va='top')
        
        y_base -= 0.05
        circle1 = plt.Circle((legend_x + 0.015, y_base), 0.012, fc='#00ff00', ec='black', linewidth=2,
                            transform=fig.transFigure)
        fig.patches.append(circle1)
        fig.text(legend_x + 0.04, y_base, 'Green', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        y_base -= 0.045
        circle2 = plt.Circle((legend_x + 0.015, y_base), 0.012, fc='#ffff00', ec='black', linewidth=2,
                            transform=fig.transFigure)
        fig.patches.append(circle2)
        fig.text(legend_x + 0.04, y_base, 'Yellow', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        y_base -= 0.045
        circle3 = plt.Circle((legend_x + 0.015, y_base), 0.012, fc='#ff0000', ec='black', linewidth=2,
                            transform=fig.transFigure)
        fig.patches.append(circle3)
        fig.text(legend_x + 0.04, y_base, 'Red', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        # queue bars section
        y_base -= 0.08
        fig.text(legend_x, y_base, '▬ Queue Length', fontsize=16, fontweight='bold', 
                transform=fig.transFigure, ha='left', va='top')
        
        y_base -= 0.05
        rect1 = plt.Rectangle((legend_x, y_base - 0.009), 0.025, 0.018, fc='#2ecc71', ec='black', linewidth=2,
                             transform=fig.transFigure)
        fig.patches.append(rect1)
        fig.text(legend_x + 0.04, y_base, '0-5 cars', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        y_base -= 0.045
        rect2 = plt.Rectangle((legend_x, y_base - 0.009), 0.025, 0.018, fc='#f39c12', ec='black', linewidth=2,
                             transform=fig.transFigure)
        fig.patches.append(rect2)
        fig.text(legend_x + 0.04, y_base, '5-10 cars', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        y_base -= 0.045
        rect3 = plt.Rectangle((legend_x, y_base - 0.009), 0.025, 0.018, fc='#e74c3c', ec='black', linewidth=2,
                             transform=fig.transFigure)
        fig.patches.append(rect3)
        fig.text(legend_x + 0.04, y_base, '>10 cars', fontsize=14, 
                transform=fig.transFigure, ha='left', va='center')
        
        # stats section
        y_base -= 0.08
        fig.text(legend_x, y_base, 'Statistics', fontsize=16, fontweight='bold', 
                transform=fig.transFigure, ha='left', va='top')
        
        y_base -= 0.045
        fig.text(legend_x, y_base, f'Time: {current_time:.0f}s', fontsize=14, 
                transform=fig.transFigure, ha='left', va='top')
        
        y_base -= 0.04
        fig.text(legend_x, y_base, f'Total Queue: {total_vehicles}', fontsize=14, 
                transform=fig.transFigure, ha='left', va='top')
        
        # title
        status_text = "FLOWING" if max_queue_current < 10 else "CONGESTED"
        ax.set_title(f'Traffic Network Animation | Time: {current_time:.0f}s ({current_time/60:.1f} min) | {status_text}',
                    fontsize=20, fontweight='bold', pad=25)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#ecf0f1')
        ax.grid(False)
    
    print(f"Creating traffic animation with {num_frames} frames...")
    print(f"Animation will be {num_frames/30:.1f} seconds long at 30 fps")
    
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                  interval=50, repeat=True, blit=False)
    
    if save_path:
        try:
            print(f"Saving animation to {save_path} (requires ffmpeg)...")
            anim.save(save_path, writer='ffmpeg', fps=30, dpi=120, bitrate=2500)
            print(f"[SUCCESS] High-quality animation saved to {save_path}")
            plt.close(fig)
        except Exception as e:
            print(f"[FAILED] Could not save animation: {e}")
            print("  Install ffmpeg: winget install ffmpeg")
    else:
        plt.show()

    return anim