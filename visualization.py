import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
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
    ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
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
    
    plt.close()


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

    plt.close()


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


def generate_all_plots(baseline_metrics: Dict, 
                       num_intersections: int,
                       output_dir: str,
                       pso_metrics: Dict = None, 
                       aco_metrics: Dict = None,
                       pso_history: Dict = None,
                       aco_history: Dict = None,
                       baseline_timings: np.ndarray = None, 
                       optimized_timings: np.ndarray = None):
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
        pso_metrics=pso_metrics if pso_metrics is not None else baseline_metrics,
        aco_metrics=aco_metrics,
        save_path=os.path.join(output_dir, 'comparison.png')
    )
    
    # 3. improvement percentages
    print("Generating improvement plot...")
    plot_improvement_percentages(
        baseline_metrics=baseline_metrics,
        pso_metrics=pso_metrics if pso_metrics is not None else baseline_metrics,
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
    
    print(f"All plots saved to '{output_dir}/'")
    print("~" * 60 + "\n")


def plot_pso_particles_live(pso, iteration: int):
    """
    Real-time visualization of PSO particles (fast, non-blocking version).
    Shows first 2 decision variables (Intersection 0: NS green, EW green).
    """
    # create figure on first call only
    if not hasattr(plot_pso_particles_live, 'fig'):
        plt.ion()  # interactive mode
        plot_pso_particles_live.fig, plot_pso_particles_live.axes = plt.subplots(1, 2, figsize=(15, 6))
        plt.show(block=False)
    
    fig = plot_pso_particles_live.fig
    ax1, ax2 = plot_pso_particles_live.axes
    
    ax1.clear()
    ax2.clear()
    
    # --- LEFT PLOT: Particle positions (simple scatter, no contour) ---
    
    # extract first 2 dimensions (Intersection 0: NS green, EW green)
    positions_2d = pso.positions[:, :2]
    pbest_2d = pso.personal_best_positions[:, :2]
    gbest_2d = pso.global_best_position[:2] if pso.global_best_position is not None else None
    
    # color particles by their fitness (darker = better)
    particle_scores = [pso.personal_best_scores[i] for i in range(pso.num_particles)]
    
    # plot particles with color mapping
    scatter = ax1.scatter(positions_2d[:, 0], positions_2d[:, 1], 
                         c=particle_scores, cmap='RdYlGn_r', s=150, alpha=0.8, 
                         edgecolors='black', linewidth=1.5, vmin=min(particle_scores), vmax=max(particle_scores),
                         label=f'Particles (n={pso.num_particles})', zorder=5)
    
    # add colorbar
    if iteration == 0:
        plot_pso_particles_live.cbar = plt.colorbar(scatter, ax=ax1)
        plot_pso_particles_live.cbar.set_label('Fitness Score', fontsize=10)
    else:
        plot_pso_particles_live.cbar.update_normal(scatter)
    
    # plot personal bests (smaller, transparent)
    ax1.scatter(pbest_2d[:, 0], pbest_2d[:, 1], 
               c='green', s=60, alpha=0.4, marker='x',
               label='Personal Bests', zorder=4)
    
    # plot global best (big red star)
    if gbest_2d is not None:
        ax1.scatter(gbest_2d[0], gbest_2d[1], 
                   c='red', s=400, alpha=1.0, marker='*', edgecolors='black', linewidth=2,
                   label=f'Global Best ({pso.global_best_score:.2f})', zorder=10)
    
    # draw velocity vectors (arrows)
    velocities_2d = pso.velocities[:, :2]
    for i in range(len(positions_2d)):
        # only draw if velocity is non-negligible
        vel_mag = np.linalg.norm(velocities_2d[i])
        if vel_mag > 0.1:  # skip tiny velocities
            ax1.arrow(positions_2d[i, 0], positions_2d[i, 1],
                    velocities_2d[i, 0] * 0.3, velocities_2d[i, 1] * 0.3,  # reduced scaling
                    head_width=0.4, head_length=0.3,  # smaller arrow head
                    fc='cyan', ec='darkblue', 
                    alpha=0.6, linewidth=0.8, zorder=3)
    
    ax1.set_xlabel('Intersection 0: NS Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Intersection 0: EW Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'PSO Swarm - Iteration {iteration + 1}/{pso.num_iterations}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    x_min = min(positions_2d[:, 0].min(), pbest_2d[:, 0].min())
    x_max = max(positions_2d[:, 0].max(), pbest_2d[:, 0].max())
    y_min = min(positions_2d[:, 1].min(), pbest_2d[:, 1].min())
    y_max = max(positions_2d[:, 1].max(), pbest_2d[:, 1].max())

    padding = 5
    ax1.set_xlim(x_min - padding, x_max + padding)
    ax1.set_ylim(y_min - padding, y_max + padding)
    
    # add search bounds rectangle
    rect = plt.Rectangle((pso.bounds[0], pso.bounds[0]), 
                         pso.bounds[1] - pso.bounds[0], 
                         pso.bounds[1] - pso.bounds[0],
                         fill=False, edgecolor='gray', linewidth=2, linestyle='--', zorder=1)
    ax1.add_patch(rect)
    
    # --- RIGHT PLOT: Convergence curve ---
    
    iterations = pso.history['iterations']
    best_scores = pso.history['best_scores']
    avg_scores = pso.history['avg_scores']
    
    ax2.plot(iterations, best_scores, 'b-', linewidth=2.5, label='Best Score', 
            marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='white')
    ax2.plot(iterations, avg_scores, 'orange', linewidth=2, linestyle='--', alpha=0.8, 
            label='Average Score', marker='s', markersize=4)
    
    ax2.fill_between(iterations, best_scores, avg_scores, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Objective Function Value', fontsize=11, fontweight='bold')
    ax2.set_title('Convergence Progress', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # annotate current best
    if len(best_scores) > 0:
        improvement = ((best_scores[0] - best_scores[-1]) / best_scores[0] * 100) if best_scores[0] > 0 else 0
        ax2.text(0.02, 0.98, 
                f'Current Best: {pso.global_best_score:.2f}\nImprovement: {improvement:.1f}%',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # save frame every iteration (for convergence animation)
    import config
    save_convergence_frame(fig, 'pso', iteration, config.OUTPUT_DIR)

    if iteration == pso.num_iterations - 1:  # last iteration
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        
        save_path = os.path.join(config.OUTPUT_DIR, 'pso_particles_final.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Final PSO particle plot saved to {save_path}")

        # stitch all frames into MP4
        stitch_convergence_animation('pso', config.OUTPUT_DIR)


def plot_aco_archive_live(aco, iteration: int):
    """
    Real-time visualization of ACO archive (mirrors plot_pso_particles_live style).
    Shows first 2 decision variables (Intersection 0: NS green, EW green).
    """
    # create figure on first call only
    if not hasattr(plot_aco_archive_live, 'fig'):
        plt.ion()
        plot_aco_archive_live.fig, plot_aco_archive_live.axes = plt.subplots(1, 2, figsize=(15, 6))
        plt.show(block=False)

    fig = plot_aco_archive_live.fig
    ax1, ax2 = plot_aco_archive_live.axes

    ax1.clear()
    ax2.clear()

    # --- LEFT PLOT: Archive solutions scatter ---

    # extract archive data
    archive_scores = [score for score, _ in aco.archive]
    archive_solutions = np.array([sol for _, sol in aco.archive])

    # extract first 2 dimensions (Intersection 0: NS green, EW green)
    if len(archive_solutions) > 0:
        solutions_2d = archive_solutions[:, :2]

        # color by fitness (darker = better, rank 0 = best)
        scatter = ax1.scatter(
            solutions_2d[:, 0], solutions_2d[:, 1],
            c=archive_scores, cmap='RdYlGn_r', s=200, alpha=0.85,
            edgecolors='black', linewidth=1.5,
            vmin=min(archive_scores), vmax=max(archive_scores),
            label=f'Archive (k={aco.archive_size})', zorder=5
        )

        # colorbar
        if iteration == 0:
            plot_aco_archive_live.cbar = plt.colorbar(scatter, ax=ax1)
            plot_aco_archive_live.cbar.set_label('Fitness Score', fontsize=10)
        else:
            plot_aco_archive_live.cbar.update_normal(scatter)

        # annotate rank on each solution
        for rank, (sol, score) in enumerate(zip(solutions_2d, archive_scores)):
            ax1.annotate(f'#{rank+1}', (sol[0], sol[1]),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=8, color='black', fontweight='bold')

        # highlight global best (rank 1 = top of archive)
        best_sol = solutions_2d[0]
        ax1.scatter(best_sol[0], best_sol[1],
                   c='red', s=450, alpha=1.0, marker='*',
                   edgecolors='black', linewidth=2,
                   label=f'Best (score={aco.global_best_score:.2f})', zorder=10)

        # draw weights as circle sizes (bigger circle = higher weight)
        weights = aco._compute_weights()
        for i, (sol, w) in enumerate(zip(solutions_2d, weights)):
            ax1.add_patch(plt.Circle(
                (sol[0], sol[1]), radius=w * 15,
                fill=False, edgecolor='blue', linewidth=1.5,
                alpha=0.4, linestyle='--', zorder=3
            ))

    ax1.set_xlabel('Intersection 0: NS Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Intersection 0: EW Green Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'ACO Archive - Iteration {iteration + 1}/{aco.num_iterations}',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # search bounds rectangle
    rect = plt.Rectangle(
        (aco.bounds[0], aco.bounds[0]),
        aco.bounds[1] - aco.bounds[0],
        aco.bounds[1] - aco.bounds[0],
        fill=False, edgecolor='gray', linewidth=2, linestyle='--', zorder=1
    )
    ax1.add_patch(rect)

    padding = 5
    ax1.set_xlim(aco.bounds[0] - padding, aco.bounds[1] + padding)
    ax1.set_ylim(aco.bounds[0] - padding, aco.bounds[1] + padding)

    # --- RIGHT PLOT: Convergence curve (identical style to PSO) ---

    iterations = aco.history['iterations']
    best_scores = aco.history['best_scores']
    avg_scores = aco.history['avg_scores']

    ax2.plot(iterations, best_scores, 'b-', linewidth=2.5, label='Best Score',
            marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='white')
    ax2.plot(iterations, avg_scores, 'orange', linewidth=2, linestyle='--', alpha=0.8,
            label='Average Score', marker='s', markersize=4)

    ax2.fill_between(iterations, best_scores, avg_scores, alpha=0.2, color='blue')

    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Objective Function Value', fontsize=11, fontweight='bold')
    ax2.set_title('Convergence Progress', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # annotate current best (identical to PSO version)
    if len(best_scores) > 0:
        improvement = ((best_scores[0] - best_scores[-1]) / best_scores[0] * 100) if best_scores[0] > 0 else 0
        ax2.text(0.02, 0.98,
                f'Current Best: {aco.global_best_score:.2f}\nImprovement: {improvement:.1f}%',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # save frame every iteration (for convergence animation)
    import config
    save_convergence_frame(fig, 'aco', iteration, config.OUTPUT_DIR)

    # save on last iteration
    if iteration == aco.num_iterations - 1:
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)

        save_path = os.path.join(config.OUTPUT_DIR, 'aco_archive_final.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] Final ACO archive plot saved to {save_path}")

        # stitch all frames into MP4
        stitch_convergence_animation('aco', config.OUTPUT_DIR)


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
    Create professional animated visualization with MASSIVE spacing.
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