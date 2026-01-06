import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Dict, List
import os


def plot_optimization_convergence(pso_history: Dict, aco_history: Dict = None, 
                                   save_path: str = None):
    """
    Plot convergence curves for PSO (and optionally ACO).
    Shows how objective function improves over iterations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSO convergence
    ax1.plot(pso_history['iterations'], pso_history['best_scores'], 
             'b-', linewidth=2, label='Best Score')
    ax1.plot(pso_history['iterations'], pso_history['avg_scores'], 
             'b--', alpha=0.6, label='Average Score')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective Function Value', fontsize=12)
    ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TODO: add ACO convergence if provided
    # comparison plot if only PSO
    ax2.plot(pso_history['iterations'], pso_history['best_scores'], 
                'g-', linewidth=2, label='PSO Best')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Objective Function Value', fontsize=12)
    ax2.set_title('Best Solution Progress', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    plt.show()


def plot_comparison_bars(baseline_metrics: Dict, pso_metrics: Dict, 
                        aco_metrics: Dict = None, save_path: str = None):
    """
    Bar chart comparing baseline vs optimized performance.
    """
    metrics_to_compare = ['avg_waiting_time', 'max_queue_length']
    metric_labels = ['Avg Waiting Time (s)', 'Max Queue Length']
    
    x = np.arange(len(metrics_to_compare))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_values = [baseline_metrics[m] for m in metrics_to_compare]
    pso_values = [pso_metrics[m] for m in metrics_to_compare]
    
    bars1 = ax.bar(x - width, baseline_values, width, label='Baseline', 
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, pso_values, width, label='PSO Optimized', 
                   color='#3498db', alpha=0.8)
    
    if aco_metrics is not None:
        aco_values = [aco_metrics[m] for m in metrics_to_compare]
        bars3 = ax.bar(x + width, aco_values, width, label='ACO Optimized', 
                      color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Optimized', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    if aco_metrics is not None:
        add_labels(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def plot_improvement_percentages(baseline_metrics: Dict, pso_metrics: Dict, 
                                 aco_metrics: Dict = None, save_path: str = None):
    """
    Show percentage improvement from baseline.
    """
    metrics = ['avg_waiting_time', 'max_queue_length']
    metric_labels = ['Waiting Time', 'Queue Length']
    
    # calculate improvements
    pso_improvements = []
    for m in metrics:
        improvement = (baseline_metrics[m] - pso_metrics[m]) / baseline_metrics[m] * 100
        pso_improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x, pso_improvements, width, label='PSO', 
                   color='#3498db', alpha=0.8)
    
    if aco_metrics is not None:
        aco_improvements = []
        for m in metrics:
            improvement = (baseline_metrics[m] - aco_metrics[m]) / baseline_metrics[m] * 100
            aco_improvements.append(improvement)
        
        bars2 = ax.bar(x + width, aco_improvements, width, label='ACO', 
                      color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Improvement over Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    if aco_metrics is not None:
        add_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement plot to {save_path}")
    
    plt.show()


def plot_signal_timings(baseline_timings: np.ndarray, optimized_timings: np.ndarray,
                       num_intersections: int, save_path: str = None):
    """
    Compare baseline vs optimized signal timings for each intersection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
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
    
    plt.suptitle('Traffic Signal Timing Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved signal timing plot to {save_path}")
    
    plt.show()


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
    
    plt.show()


def create_traffic_animation(network, simulation_data: Dict, save_path: str = None):
    """
    Create animated visualization of traffic flow through network.
    Shows queue lengths, traffic light states, and vehicle flow over time.
    
    Args:
        network: Network object with intersections
        simulation_data: Dict with 'queue_samples' = {lane_id: [(time, queue_length), ...]}
                        and 'light_states' = {int_id: [(time, phase), ...]}
        save_path: Path to save animation (requires ffmpeg)
    
    Returns:
        Animation object
    """
    import config
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # get intersection positions from network
    positions = network.get_positions()

    # extract queue data
    queue_samples = simulation_data.get('queue_samples', {})
    
    # get max time from queue samples
    max_time = 0
    for lane_id, samples in queue_samples.items():
        if len(samples) > 0:
            max_time = max(max_time, max([t for t, q in samples]))
    
    if max_time == 0:
        max_time = config.SIMULATION_DURATION
    
    # calculate number of frames (sample every 10 seconds)
    time_step = 10  # seconds per frame
    num_frames = int(max_time / time_step)
    
    def get_queue_at_time(lane_id, time):
        """Interpolate queue length at specific time"""
        if lane_id not in queue_samples or len(queue_samples[lane_id]) == 0:
            return 0
        
        samples = queue_samples[lane_id]
        
        # find nearest sample
        for i, (t, q) in enumerate(samples):
            if t >= time:
                if i == 0:
                    return q
                # linear interpolation
                t_prev, q_prev = samples[i - 1]
                ratio = (time - t_prev) / (t - t_prev) if t != t_prev else 0
                return q_prev + ratio * (q - q_prev)
        
        # return last value if time exceeds samples
        return samples[-1][1] if samples else 0
    
    def get_light_phase_at_time(int_id, time):
        """Get traffic light phase at specific time from simulation data"""
        if 'light_states' in simulation_data and int_id in simulation_data['light_states']:
            states = simulation_data['light_states'][int_id]
            
            # find the phase at this time
            for i, (t, phase) in enumerate(states):
                if t >= time:
                    return states[i-1][1] if i > 0 else phase
            
            return states[-1][1] if states else 0
        
        # fallback: calculate from cycle time (what we have now)
        intersection = network.intersections[int_id]
        cycle_time = intersection.traffic_light.cycle_time
        time_in_cycle = time % cycle_time
        
        if time_in_cycle < intersection.traffic_light.green_ns:
            return 0  # NS green
        elif time_in_cycle < intersection.traffic_light.green_ns + intersection.traffic_light.yellow:
            return 1  # NS yellow
        elif time_in_cycle < intersection.traffic_light.green_ns + intersection.traffic_light.yellow + intersection.traffic_light.green_ew:
            return 2  # EW green
        else:
            return 3  # EW yellow
    
    def update(frame):
        """Update function for each animation frame"""
        ax.clear()
        
        current_time = frame * time_step
        
        # set plot limits dynamically based on actual positions
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]

        x_min, x_max = min(x_coords) - 0.5, max(x_coords) + 0.5
        y_min, y_max = min(y_coords) - 0.5, max(y_coords) + 0.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f'Traffic Network Animation - Time: {current_time:.0f}s / {max_time:.0f}s', 
                    fontsize=14, fontweight='bold')
        
        # draw roads (connections between intersections)
        for int_id, intersection in network.intersections.items():
            x1, y1 = positions[int_id]
            
            for direction, connected_id in intersection.outgoing_connections.items():
                if connected_id in positions:
                    x2, y2 = positions[connected_id]
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=8, alpha=0.3, zorder=1)
        
        # draw queue lengths on lanes
        for int_id, intersection in network.intersections.items():
            x, y = positions[int_id]
            
            for direction, lane in intersection.lanes.items():
                queue_length = get_queue_at_time(lane.lane_id, current_time)
                
                # normalize queue length to bar size (0-0.3 units)
                max_queue_visual = 0.3
                queue_visual = min(queue_length / config.MAX_QUEUE_THRESHOLD, 1.0) * max_queue_visual
                
                # direction offsets
                offsets = {
                    'N': (0, max_queue_visual),
                    'S': (0, -max_queue_visual),
                    'E': (max_queue_visual, 0),
                    'W': (-max_queue_visual, 0)
                }
                
                dx, dy = offsets[direction]
                
                # color based on queue severity
                if queue_length < config.MAX_QUEUE_THRESHOLD * 0.5:
                    color = '#2ecc71'  # Green - light traffic
                elif queue_length < config.MAX_QUEUE_THRESHOLD * 0.8:
                    color = '#f39c12'  # Orange - moderate traffic
                else:
                    color = '#e74c3c'  # Red - heavy traffic
                
                # draw queue bar
                if direction in ['N', 'S']:
                    # vertical bar
                    rect = plt.Rectangle((x - 0.05, y), 0.1, dy * (queue_visual / max_queue_visual), 
                                        color=color, alpha=0.7, zorder=2)
                else:
                    # horizontal bar
                    rect = plt.Rectangle((x, y - 0.05), dx * (queue_visual / max_queue_visual), 0.1, 
                                        color=color, alpha=0.7, zorder=2)
                
                ax.add_patch(rect)
                
                # add queue length text
                text_offset = 1.5
                tx = x + dx * text_offset
                ty = y + dy * text_offset
                ax.text(tx, ty, f'{int(queue_length)}', 
                       fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # draw intersections with traffic light colors
        for int_id, (x, y) in positions.items():
            phase = get_light_phase_at_time(int_id, current_time)
            
            # choose color based on phase
            if phase == 0:
                light_color = '#2ecc71'  # NS green
            elif phase == 1:
                light_color = '#f39c12'  # NS yellow
            elif phase == 2:
                light_color = '#3498db'  # EW green (different shade)
            else:
                light_color = '#f39c12'  # EW yellow
            
            circle = plt.Circle((x, y), 0.15, color=light_color, 
                              edgecolor='black', linewidth=2, alpha=0.9, zorder=10)
            ax.add_patch(circle)
            
            ax.text(x, y, str(int_id), color='white', 
                   ha='center', va='center', fontweight='bold', 
                   fontsize=12, zorder=11)
        
        # add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='#2ecc71', label='Light Traffic'),
            plt.Rectangle((0, 0), 1, 1, fc='#f39c12', label='Moderate Traffic'),
            plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', label='Heavy Traffic')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#f0f0f0')
        
        # add grid
        ax.grid(True, alpha=0.2, linestyle='--')
    
    # create animation
    print(f"Creating animation with {num_frames} frames...")
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                  interval=config.ANIMATION_INTERVAL, 
                                  repeat=True, blit=False)
    
    if save_path:
        try:
            print(f"Attempting to save animation to {save_path}...")
            print("NOTE: This requires ffmpeg to be installed and in PATH.")
            anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
            print(f"[SUCCESS] Animation saved successfully to {save_path}")
        except Exception as e:
            print(f"[FAILED] Failed to save animation: {e}")
            print("  Fallback: Showing animation instead.")
            print("  To save animations, install ffmpeg:")
            print("    - Windows: Download from https://ffmpeg.org/download.html")
            print("    - Or run: winget install ffmpeg")
            plt.show()
    else:
        plt.show()

    return anim