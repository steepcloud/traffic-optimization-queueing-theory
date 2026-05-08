import numpy as np
import time
import config
from typing import Dict, Tuple, Callable, Optional, List
from multiprocessing import Pool
import os

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


def dummy_objective(timings: np.ndarray) -> float:
    """Simple test function: minimize sum of squares around 50."""
    return np.sum((timings - 50) ** 2)


class PSO:
    """
    Particle Swarm Optimization for traffic signal timing.

    Each particle represents a candidate set of green light durations.
    Fitness is evaluated by running the traffic simulation.
    """

    def __init__(self, objective_function: Callable, num_variables: int,
                 bounds: Tuple[float, float], config: Dict,
                 num_processes: Optional[int] = None):
        """
        Args:
            objective_function: Callable(timings) -> float, lower is better.
            num_variables: Decision variables (2 per intersection: NS green, EW green).
            bounds: (min, max) for every variable.
            config: PSO hyperparameters — num_particles, num_iterations, w, c1, c2.
            num_processes: Parallel workers (default: os.cpu_count()).
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = bounds

        self.num_particles  = config['num_particles']
        self.num_iterations = config['num_iterations']
        self.w  = config['w']
        self.c1 = config['c1']
        self.c2 = config['c2']

        self.positions = np.random.uniform(bounds[0], bounds[1],
                                           (self.num_particles, num_variables))
        self.velocities = np.random.uniform(-1, 1,
                                            (self.num_particles, num_variables))

        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.num_particles, float('inf'))

        self.global_best_position = None
        self.global_best_score = float('inf')

        self.history = {'best_scores': [], 'avg_scores': [], 'iterations': []}
        self.num_processes = num_processes or os.cpu_count()

        self.patience = 10
        self.min_improvement = 0.005
        self.best_score_history: List[float] = []

    def _check_early_stop(self, verbose: int) -> bool:
        """
        Return True if improvement over the last `patience` iterations
        is below the minimum threshold.
        """
        if len(self.best_score_history) <= self.patience:
            return False

        recent_improvement = (
            self.best_score_history[-self.patience] - self.global_best_score
        ) / self.best_score_history[-self.patience]

        if recent_improvement < self.min_improvement:
            if verbose >= 1:
                print(f"\n[EARLY STOP] No significant improvement for {self.patience} iterations")
                print(f"Recent improvement: {recent_improvement*100:.2f}% < {self.min_improvement*100:.2f}% threshold")
            return True

        return False

    def optimize(self, verbose: int = 1) -> Tuple[np.ndarray, float]:
        """
        Run PSO and return (best_timings, best_score).
        """
        start_time = time.time()

        if verbose >= 1:
            print(f"{'~' * 5} PARTICLE SWARM OPTIMIZATION {'~' * 5}")
            print(f"Particles: {self.num_particles}")
            print(f"Iterations: {self.num_iterations}")
            print(f"Variables: {self.num_variables}")
            print(f"Bounds: {self.bounds}")
            print(f"{'~' * 60}\n")

        for iteration in range(self.num_iterations):
            iteration_start = time.time()

            with Pool(processes=self.num_processes) as pool:
                scores = pool.map(self.objective_function, self.positions)

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i].copy()

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()

            for i in range(self.num_particles):
                r1 = np.random.random(self.num_variables)
                r2 = np.random.random(self.num_variables)

                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social    = self.c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i], self.bounds[0], self.bounds[1]
                )

            avg_score = np.mean(scores)
            self.history['best_scores'].append(self.global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['iterations'].append(iteration)
            self.best_score_history.append(self.global_best_score)

            should_stop = self._check_early_stop(verbose)

            if verbose >= 1:
                if config.SHOW_PLOTS_DURING_OPT:
                    from visualization import plot_pso_particles_live
                    is_final = (iteration == self.num_iterations - 1) or should_stop
                    plot_pso_particles_live(self, iteration, is_final=is_final)

                print(f"Iteration {iteration + 1:3d}/{self.num_iterations} | "
                      f"Best: {self.global_best_score:8.2f} | "
                      f"Avg: {avg_score:8.2f} | "
                      f"Time: {time.time() - iteration_start:5.1f}s", flush=True)

            if should_stop:
                break

        if verbose >= 1:
            print(f"\n{'~' * 60}")
            print(f"PSO Complete! Total time: {time.time() - start_time:.1f}s")
            print(f"Best objective: {self.global_best_score:.2f}")
            print(f"Best timings: {self.global_best_position}")
            print(f"{'~' * 60}\n")

        return self.global_best_position, self.global_best_score

    def get_history(self) -> Dict:
        """Return optimization history for plotting."""
        return self.history


class ACO:
    """
    Ant Colony Optimization for Continuous Domains (ACOR) for traffic signal timing.

    Maintains a sorted archive of the best solutions found. Each ant samples
    a new solution from the archive using weighted Gaussian perturbation.
    """

    def __init__(self, objective_function: Callable, num_variables: int,
                 bounds: Tuple[float, float], config: Dict,
                 num_processes: Optional[int] = None):
        """
        Args:
            objective_function: Callable(timings) -> float, lower is better.
            num_variables: Decision variables (2 per intersection: NS green, EW green).
            bounds: (min, max) for every variable.
            config: ACO hyperparameters — n_ants, archive_size, q, xi, num_iterations.
            num_processes: Parallel workers (default: os.cpu_count()).
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = bounds

        self.n_ants       = config['n_ants']
        self.archive_size = config['archive_size']
        self.q            = config['q']
        self.xi           = config['xi']
        self.num_iterations = config['num_iterations']

        self.archive: List[Tuple[float, np.ndarray]] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_score: float = float('inf')

        self.history = {'best_scores': [], 'avg_scores': [], 'iterations': []}
        self.num_processes = num_processes or os.cpu_count()

        self.patience = 10
        self.min_improvement = 0.005
        self.best_score_history: List[float] = []

    def _check_early_stop(self, verbose: int) -> bool:
        """
        Return True if improvement over the last `patience` iterations
        is below the minimum threshold.
        """
        if len(self.best_score_history) <= self.patience:
            return False

        recent_improvement = (
            self.best_score_history[-self.patience] - self.global_best_score
        ) / self.best_score_history[-self.patience]

        if recent_improvement < self.min_improvement:
            if verbose >= 1:
                print(f"\n[EARLY STOP] No significant improvement for {self.patience} iterations")
                print(f"Recent improvement: {recent_improvement*100:.2f}% < {self.min_improvement*100:.2f}% threshold")
            return True

        return False

    def _compute_weights(self) -> np.ndarray:
        """
        Gaussian weights for each archive solution, rank 1 (best) gets highest weight.
        Returns a normalized array summing to 1.
        """
        k = self.archive_size
        ranks = np.arange(1, k + 1)
        weights = (1 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(
            -(ranks - 1) ** 2 / (2 * (self.q * k) ** 2)
        )
        return weights / weights.sum()

    def _sample_solution(self) -> np.ndarray:
        """
        Construct a new solution by picking a template from the archive
        (weighted random) and adding Gaussian noise scaled by xi.
        """
        weights = self._compute_weights()
        archive_solutions = [sol for _, sol in self.archive]

        selected_idx = np.random.choice(len(self.archive), p=weights)
        template = archive_solutions[selected_idx]

        new_solution = np.zeros(self.num_variables)
        for d in range(self.num_variables):
            sigma = (
                self.xi * sum(abs(archive_solutions[j][d] - template[d])
                              for j in range(len(self.archive)))
                / (self.archive_size - 1)
            ) if self.archive_size > 1 else self.xi

            new_solution[d] = template[d] + np.random.normal(0, sigma + 1e-10)

        return np.clip(new_solution, self.bounds[0], self.bounds[1])

    def _update_archive(self, new_solutions: List[Tuple[float, np.ndarray]]) -> None:
        """
        Merge new solutions into archive, keeping only the best archive_size entries.
        Updates global best if improved.
        """
        self.archive.extend(new_solutions)
        self.archive.sort(key=lambda x: x[0])
        self.archive = self.archive[:self.archive_size]

        best_score, best_solution = self.archive[0]
        if best_score < self.global_best_score:
            self.global_best_score = best_score
            self.global_best_position = best_solution.copy()

    def optimize(self, verbose: int = 1) -> Tuple[np.ndarray, float]:
        """
        Run ACOR and return (best_timings, best_score).
        """
        start_time = time.time()

        if verbose >= 1:
            print(f"{'~' * 5} ANT COLONY OPTIMIZATION (ACOR) {'~' * 5}")
            print(f"Ants: {self.n_ants}")
            print(f"Archive Size: {self.archive_size}")
            print(f"Iterations: {self.num_iterations}")
            print(f"Variables: {self.num_variables}")
            print(f"Bounds: {self.bounds}")
            print(f"q (locality): {self.q} | xi (evaporation): {self.xi}")
            print(f"{'~' * 60}\n")
            print("Initializing archive with random solutions...")

        initial_solutions = np.random.uniform(
            self.bounds[0], self.bounds[1],
            (self.archive_size, self.num_variables)
        )
        with Pool(processes=self.num_processes) as pool:
            initial_scores = pool.map(self.objective_function, initial_solutions)

        self.archive = [
            (initial_scores[i], initial_solutions[i].copy())
            for i in range(self.archive_size)
        ]
        self.archive.sort(key=lambda x: x[0])
        self.global_best_score, self.global_best_position = self.archive[0]
        self.global_best_position = self.global_best_position.copy()

        if verbose >= 1:
            print(f"Archive initialized | Best initial score: {self.global_best_score:.2f}\n")

        for iteration in range(self.num_iterations):
            iteration_start = time.time()

            ant_solutions = np.array([self._sample_solution() for _ in range(self.n_ants)])

            with Pool(processes=self.num_processes) as pool:
                scores = pool.map(self.objective_function, ant_solutions)

            self._update_archive([
                (scores[i], ant_solutions[i].copy()) for i in range(self.n_ants)
            ])

            avg_score = np.mean(scores)
            self.history['best_scores'].append(self.global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['iterations'].append(iteration)
            self.best_score_history.append(self.global_best_score)

            should_stop = self._check_early_stop(verbose)

            if verbose >= 1:
                if config.SHOW_PLOTS_DURING_OPT:
                    from visualization import plot_aco_archive_live
                    is_final = (iteration == self.num_iterations - 1) or should_stop
                    plot_aco_archive_live(self, iteration, is_final=is_final)

                print(f"Iteration {iteration + 1:3d}/{self.num_iterations} | "
                      f"Best: {self.global_best_score:8.2f} | "
                      f"Avg: {avg_score:8.2f} | "
                      f"Time: {time.time() - iteration_start:5.1f}s", flush=True)

            if should_stop:
                break

        if verbose >= 1:
            print(f"\n{'~' * 60}")
            print(f"ACO Complete! Total time: {time.time() - start_time:.1f}s")
            print(f"Best objective: {self.global_best_score:.2f}")
            print(f"Best timings: {self.global_best_position}")
            print(f"{'~' * 60}\n")

        return self.global_best_position, self.global_best_score

    def get_history(self) -> Dict:
        """Return optimization history for plotting."""
        return self.history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pso_initialization():
    print("Testing PSO initialization...", end=" ")

    pso = PSO(
        objective_function=dummy_objective,
        num_variables=8,
        bounds=(30, 90),
        config={'num_particles': 10, 'num_iterations': 5, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}
    )

    assert pso.positions.shape == (10, 8)
    assert pso.velocities.shape == (10, 8)
    assert pso.global_best_score == float('inf')

    print(f"{GREEN}PASSED{RESET}")


def test_pso_optimize():
    print("Testing PSO optimization...", end=" ")

    pso = PSO(
        objective_function=dummy_objective,
        num_variables=4,
        bounds=(30, 90),
        config={'num_particles': 5, 'num_iterations': 3, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}
    )

    best_pos, best_score = pso.optimize(verbose=0)

    assert best_pos.shape == (4,)
    assert best_score < float('inf')
    assert len(pso.history['best_scores']) == 3

    print(f"{GREEN}PASSED{RESET}")


def test_aco_initialization():
    print("Testing ACO initialization...", end=" ")

    aco = ACO(
        objective_function=dummy_objective,
        num_variables=8,
        bounds=(30, 90),
        config={'n_ants': 10, 'archive_size': 10, 'q': 0.5, 'xi': 0.85, 'num_iterations': 5}
    )

    assert aco.archive == []
    assert aco.global_best_score == float('inf')
    assert aco.global_best_position is None
    assert aco.n_ants == 10

    print(f"{GREEN}PASSED{RESET}")


def test_aco_weights():
    print("Testing ACO weight computation...", end=" ")

    aco = ACO(
        objective_function=dummy_objective,
        num_variables=8,
        bounds=(30, 90),
        config={'n_ants': 10, 'archive_size': 10, 'q': 0.5, 'xi': 0.85, 'num_iterations': 5}
    )

    weights = aco._compute_weights()

    assert len(weights) == 10
    assert abs(weights.sum() - 1.0) < 1e-10
    assert weights[0] > weights[-1]

    print(f"{GREEN}PASSED{RESET}")


def test_aco_optimize():
    print("Testing ACO optimization...", end=" ")

    aco = ACO(
        objective_function=dummy_objective,
        num_variables=4,
        bounds=(30, 90),
        config={'n_ants': 5, 'archive_size': 5, 'q': 0.5, 'xi': 0.85, 'num_iterations': 3}
    )

    best_pos, best_score = aco.optimize(verbose=0)

    assert best_pos.shape == (4,)
    assert best_score < float('inf')
    assert len(aco.history['best_scores']) == 3
    assert len(aco.archive) == 5

    print(f"{GREEN}PASSED{RESET}")


if __name__ == "__main__":
    print(f"{'~' * 5} OPTIMIZATION MODULE UNIT TESTS {'~' * 5}\n")

    tests = [
        test_pso_initialization,
        test_pso_optimize,
        test_aco_initialization,
        test_aco_weights,
        test_aco_optimize,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{RED}FAILED: {e}{RESET}")
            failed += 1
        except Exception as e:
            print(f"{RED}ERROR: {e}{RESET}")
            failed += 1

    print("\n" + "~" * 60)
    if failed == 0:
        print(f"{GREEN}RESULTS: {passed} passed, {failed} failed{RESET}")
    else:
        print(f"RESULTS: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")
    print("~" * 60)

    if failed == 0:
        print(f"\n{GREEN}All tests passed! Optimization module is working correctly.{RESET}\n")
    else:
        print(f"\n{RED}{failed} test(s) failed. Fix issues above.{RESET}\n")
        exit(1)