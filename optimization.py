import numpy as np
import time
import config
from typing import Dict, Tuple, Callable, Optional, List
from multiprocessing import Pool
import os


def dummy_objective(timings: np.ndarray) -> float:
    """Simple test function: minimize sum of squares"""
    return np.sum((timings - 50) ** 2)


class PSO:
    """
    Particle Swarm Optimization (PSO) for traffic signal timing.

    Each particle = set of green light timings for all intersections
    Fitness = objective function from simulation
    """

    def __init__(self, objective_function: Callable, num_variables: int,
                 bounds: Tuple[float, float], config: Dict, num_processes: Optional[int] = None):
        """
        Args:
            objective_function: Function that takes timings and returns fitness
            num_variables: Number of decision variables (2 per intersection: NS, EW)
            bounds: (min, max) for each variable
            config: PSO parameters (num_particles, iterations, w, c1, c2)
            num_processes: Number of parallel processes (default: os.cpu_count())
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = bounds

        # PSO parameters
        self.num_particles = config['num_particles']
        self.num_iterations = config['num_iterations']
        self.w = config['w'] # inertia weight
        self.c1 = config['c1'] # cognitive coefficient
        self.c2 = config['c2'] # social coefficient

        # init swarm
        self.positions = np.random.uniform(
            bounds[0], bounds[1],
            (self.num_particles, num_variables)
        )

        self.velocities = np.random.uniform(
            -1, 1,
            (self.num_particles, num_variables)
        )

        # best positions
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.num_particles, float('inf'))

        self.global_best_position = None
        self.global_best_score = float('inf')

        # history tracking
        self.history = {
            'best_scores': [],
            'avg_scores': [],
            'iterations': []
        }

        self.num_processes = num_processes or os.cpu_count()

        # early stopping parameters
        self.patience = 5  # stop if no improvement for 5 iterations
        self.min_improvement = 0.01  # minimum 1% improvement threshold
        self.best_score_history = []
    
    def optimize(self, verbose: int = 1) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization.

        Returns:
            (best_position, best_score)
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

            # evaluate all particles in parallel
            with Pool(processes=self.num_processes) as pool:
                scores = pool.map(self.objective_function, self.positions)

            # update personal and global bests
            for i in range(self.num_particles):
                # update personal best
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i].copy()

                # update global best
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()

            # update velocities and positions
            for i in range(self.num_particles):
                # random coefficients
                r1 = np.random.random(self.num_variables)
                r2 = np.random.random(self.num_variables)

                # velocity update
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    cognitive +
                    social
                )

                # position update
                self.positions[i] += self.velocities[i]

                # enforce bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

            # record history
            avg_score = np.mean(scores)
            self.history['best_scores'].append(self.global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['iterations'].append(iteration)

            # early stopping check
            self.best_score_history.append(self.global_best_score)
            should_stop = False
            if len(self.best_score_history) > self.patience:
                recent_improvement = (self.best_score_history[-self.patience] - self.global_best_score) / self.best_score_history[-self.patience]
                if recent_improvement < self.min_improvement:
                    if verbose >= 1:
                        print(f"\n[EARLY STOP] No significant improvement for {self.patience} iterations")
                        print(f"Recent improvement: {recent_improvement*100:.2f}% < {self.min_improvement*100:.2f}% threshold")
                    should_stop = True

            # live visualization
            if verbose >= 1:
                if config.SHOW_PLOTS_DURING_OPT:
                    from visualization import plot_pso_particles_live
                    # check if this is the final iteration (either last planned or early stop)
                    is_final = (iteration == self.num_iterations - 1) or should_stop
                    plot_pso_particles_live(self, iteration, is_final=is_final)

            # progress report
            iteration_time = time.time() - iteration_start
            if verbose >= 1:
                print(f"Iteration {iteration + 1:3d}/{self.num_iterations} | "
                      f"Best: {self.global_best_score:8.2f} | "
                      f"Avg: {avg_score:8.2f} | "
                      f"Time: {iteration_time:5.1f}s", flush=True)

            # break if early stopping triggered
            if should_stop:
                break
            
        total_time = time.time() - start_time

        if verbose >= 1:
            print(f"\n{'~' * 60}")
            print(f"PSO Complete! Total time: {total_time:.1f}s")
            print(f"Best objective: {self.global_best_score:.2f}")
            print(f"Best timings: {self.global_best_position}")
            print(f"{'~' * 60}\n")

        return self.global_best_position, self.global_best_score
    
    def get_history(self) -> Dict:
        """Return optimization history for plotting"""
        return self.history


class ACO:
    """
    Ant Colony Optimization for Continuous Domains (ACOR) for traffic signal timing.
    
    Each ant = set of green light timings for all intersections
    Archive = sorted pool of best solutions found so far
    Fitness = objective function from simulation
    """

    def __init__(self, objective_function: Callable, num_variables: int,
                 bounds: Tuple[float, float], config: Dict, num_processes: Optional[int] = None):
        """
        Args:
            objective_function: Function that takes timings and returns fitness
            num_variables: Number of decision variables (2 per intersection: NS, EW)
            bounds: (min, max) for each variable
            config: ACO parameters (n_ants, archive_size, q, xi, iterations)
            num_processes: Number of parallel processes (default: os.cpu_count())
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = bounds

        # ACO parameters
        self.n_ants = config['n_ants']                      # ants per iteration (like num_particles)
        self.archive_size = config['archive_size']          # k - how many solutions to keep
        self.q = config['q']                                # locality (0.5 = balanced explore/exploit)
        self.xi = config['xi']                              # evaporation rate (0.85 = noise shrinks over time)
        self.num_iterations = config['num_iterations']      # max iterations

        # archive stores list of [fitness, solution_array] sorted best -> worst
        # starts empty, gets populated in first iteration
        self.archive: List[Tuple[float, np.ndarray]] = []

        # best solution tracking (mirros PSO global best style)
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_score: float = float('inf')

        # history tracking (same structure as PSO for compatible plotting)
        self.history = {
            'best_scores': [],
            'avg_scores': [],
            'iterations': []
        }

        self.num_processes = num_processes or os.cpu_count()

        # early stopping
        self.patience = 5
        self.min_improvement = 0.01
        self.best_score_history = []


    def _compute_weights(self) -> np.ndarray:
        """
        Compute Gaussian weights for each solution in archive.
        Best solution (rank 1) gets highest weight.
        Returns normalized array of weights (sums to 1).
        """
        k = self.archive_size
        weights = np.zeros(k)

        for i in range(k):
            rank = i + 1  # rank starts at 1
            numerator = -(rank - 1) ** 2
            denominator = 2 * (self.q ** 2) * (k ** 2)
            weights[i] = (1 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(numerator / denominator)

        # normalize so weights sum to 1 (probability distribution)
        return weights / weights.sum()


    def _sample_solution(self) -> np.ndarray:
        """
        Sample a new solution from archive (ant constructs a solution).
        
        Steps:
        1. Pick a template solution from archive (weighted random)
        2. For each dimension, compute std deviation from archive spread
        3. Add Gaussian noise scaled by xi (evaporation)
        4. Clip to bounds
        """
        weights = self._compute_weights()

        # step 1: pick template solution (weighted random pick)
        archive_scores = [score for score, _ in self.archive]
        archive_solutions = [sol for _, sol in self.archive]
        
        selected_idx = np.random.choice(len(self.archive), p=weights)
        template = archive_solutions[selected_idx]

        # step 2 & 3: for each dimension, compute sigma and sample
        new_solution = np.zeros(self.num_variables)

        for d in range(self.num_variables):
            # sigma = xi * sum of distances from selected solution to all others
            # this is the ACOR formula for standard deviation
            sigma = self.xi * np.sum([
                abs(archive_solutions[j][d] - template[d])
                for j in range(len(self.archive))
            ]) / (self.archive_size - 1) if self.archive_size > 1 else self.xi

            # sample from Gaussian centered on template
            new_solution[d] = template[d] + np.random.normal(0, sigma + 1e-10)

        # step 4: clip to bounds
        new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])

        return new_solution
    

    def _update_archive(self, new_solutions: List[Tuple[float, np.ndarray]]) -> None:
        """
        Add new solutions to archive, keep only best k.
        
        Args:
            new_solutions: List of (fitness, solution) tuples from current iteration
        """
        # add all new solutions to archive
        self.archive.extend(new_solutions)

        # sort by fitness (best = lowest score first)
        self.archive.sort(key=lambda x: x[0])

        # keep only best k solutions
        self.archive = self.archive[:self.archive_size]

        # update global best from top of archive
        best_score, best_solution = self.archive[0]
        if best_score < self.global_best_score:
            self.global_best_score = best_score
            self.global_best_position = best_solution.copy()


    def optimize(self, verbose: int = 1) -> Tuple[np.ndarray, float]:
        """
        Run ACO optimization.

        Returns:
            (best_position, best_score)
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

        # --- PHASE 1: Initialize archive with random solutions ---
        if verbose >= 1:
            print("Initializing archive with random solutions...")

        initial_solutions = np.random.uniform(
            self.bounds[0], self.bounds[1],
            (self.archive_size, self.num_variables)
        )

        # evaluate initial solutions in parallel
        with Pool(processes=self.num_processes) as pool:
            initial_scores = pool.map(self.objective_function, initial_solutions)

        # populate archive
        self.archive = [
            (initial_scores[i], initial_solutions[i].copy())
            for i in range(self.archive_size)
        ]
        self.archive.sort(key=lambda x: x[0])  # sort best -> worst

        # set initial global best
        self.global_best_score, self.global_best_position = self.archive[0]
        self.global_best_position = self.global_best_position.copy()

        if verbose >= 1:
            print(f"Archive initialized | Best initial score: {self.global_best_score:.2f}\n")

        # --- PHASE 2: Main optimization loop ---
        for iteration in range(self.num_iterations):
            iteration_start = time.time()

            # each ant samples a new solution from archive
            ant_solutions = np.array([
                self._sample_solution() for _ in range(self.n_ants)
            ])

            # evaluate all ant solutions in parallel
            with Pool(processes=self.num_processes) as pool:
                scores = pool.map(self.objective_function, ant_solutions)

            # update archive with new solutions
            new_solutions = [
                (scores[i], ant_solutions[i].copy())
                for i in range(self.n_ants)
            ]
            self._update_archive(new_solutions)

            # record history
            avg_score = np.mean(scores)
            self.history['best_scores'].append(self.global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['iterations'].append(iteration)

            # early stopping
            self.best_score_history.append(self.global_best_score)
            should_stop = False
            if len(self.best_score_history) > self.patience:
                recent_improvement = (
                    self.best_score_history[-self.patience] - self.global_best_score
                ) / self.best_score_history[-self.patience]

                if recent_improvement < self.min_improvement:
                    if verbose >= 1:
                        print(f"\n[EARLY STOP] No significant improvement for {self.patience} iterations")
                        print(f"Recent improvement: {recent_improvement*100:.2f}% < {self.min_improvement*100:.2f}% threshold")
                    should_stop = True

            # progress report
            iteration_time = time.time() - iteration_start
            if verbose >= 1:
                if config.SHOW_PLOTS_DURING_OPT:
                    from visualization import plot_aco_archive_live
                    # check if this is the final iteration (either last planned or early stop)
                    is_final = (iteration == self.num_iterations - 1) or should_stop
                    plot_aco_archive_live(self, iteration, is_final=is_final)

                print(f"Iteration {iteration + 1:3d}/{self.num_iterations} | "
                      f"Best: {self.global_best_score:8.2f} | "
                      f"Avg: {avg_score:8.2f} | "
                      f"Time: {iteration_time:5.1f}s", flush=True)

            # break if early stopping triggered
            if should_stop:
                break

        total_time = time.time() - start_time

        if verbose >= 1:
            print(f"\n{'~' * 60}")
            print(f"ACO Complete! Total time: {total_time:.1f}s")
            print(f"Best objective: {self.global_best_score:.2f}")
            print(f"Best timings: {self.global_best_position}")
            print(f"{'~' * 60}\n")

        return self.global_best_position, self.global_best_score

    
    def get_history(self) -> Dict:
        """Return optimization history for plotting"""
        return self.history


if __name__ == "__main__":
    """Unit tests for optimization module"""
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def test_pso_initialization():
        """Test PSO initialization"""
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
        """Test PSO optimization (small run)"""
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
        """Test ACO initialization"""
        print("Testing ACO initialization...", end=" ")

        aco = ACO(
            objective_function=dummy_objective,
            num_variables=8,
            bounds=(30, 90),
            config={'n_ants': 10, 'archive_size': 10, 'q': 0.5, 
                    'xi': 0.85, 'num_iterations': 5}
        )

        assert aco.archive == []
        assert aco.global_best_score == float('inf')
        assert aco.global_best_position is None
        assert aco.n_ants == 10

        print(f"{GREEN}PASSED{RESET}")
    
    def test_aco_weights():
        """Test ACO weight computation"""
        print("Testing ACO weight computation...", end=" ")

        aco = ACO(
            objective_function=dummy_objective,
            num_variables=8,
            bounds=(30, 90),
            config={'n_ants': 10, 'archive_size': 10, 'q': 0.5,
                    'xi': 0.85, 'num_iterations': 5}
        )

        weights = aco._compute_weights()

        assert len(weights) == 10                   # one per archive slot
        assert abs(weights.sum() - 1.0) < 1e-10     # must sum to 1
        assert weights[0] > weights[-1]             # best gets highest weight

        print(f"{GREEN}PASSED{RESET}")
    
    def test_aco_optimize():
        """Test ACO optimization (small run)"""
        print("Testing ACO optimization...", end=" ")

        aco = ACO(
            objective_function=dummy_objective,
            num_variables=4,
            bounds=(30, 90),
            config={'n_ants': 5, 'archive_size': 5, 'q': 0.5,
                    'xi': 0.85, 'num_iterations': 3}
        )

        best_pos, best_score = aco.optimize(verbose=0)

        assert best_pos.shape == (4,)
        assert best_score < float('inf')
        assert len(aco.history['best_scores']) == 3
        assert len(aco.archive) == 5                       # archive stays at archive_size

        print(f"{GREEN}PASSED{RESET}")
    
    print(f"{'~' * 5} OPTIMIZATION MODULE UNIT TESTS {'~' * 5}\n")
    
    tests = [
        test_pso_initialization,
        test_pso_optimize,
        test_aco_initialization,
        test_aco_weights,
        test_aco_optimize
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
        print(f"{GREEN}RESULTS: {passed} passed, {RED}{failed} failed{RESET}")
    else:
        print(f"RESULTS: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")
    print("~" * 60)
    
    if failed == 0:
        print(f"\n{GREEN}All tests passed! Optimization module is working correctly.{RESET}\n")
    else:
        print(f"\n{RED}{failed} test(s) failed. Fix issues above.{RESET}\n")
        exit(1)