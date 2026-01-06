import numpy as np
import time
from typing import Dict, Tuple, Callable


class PSO:
    """
    Particle Swarm Optimization (PSO) for traffic signal timing.

    Each particle = set of green light timings for all intersections
    Fitness = objective function from simulation
    """

    def __init__(self, objective_function: Callable, num_variables: int,
                 bounds: Tuple[float, float], config: Dict):
        """
        Args:
            objective_function: Function that takes timings and returns fitness
            num_variables: Number of decision variables (2 per intersection: NS, EW)
            bounds: (min, max) for each variable
            config: PSO parameters (num_particles, iterations, w, c1, c2)
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
            scores = []

            # evaluate all particles
            for i in range(self.num_particles):
                score = self.objective_function(self.positions[i])
                scores.append(score)

                # update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                # update global best
                if score < self.global_best_score:
                    self.global_best_score = score
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
            
            # progress report
            iteration_time = time.time() - iteration_start
            if verbose >= 1:
                print(f"Iteration {iteration + 1:3d}/{self.num_iterations} | "
                      f"Best: {self.global_best_score:8.2f} | "
                      f"Avg: {avg_score:8.2f} | "
                      f"Time: {iteration_time:5.1f}s")
            
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


if __name__ == "__main__":
    """Unit tests for optimization module"""
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def dummy_objective(timings: np.ndarray) -> float:
        """Simple test function: minimize sum of squares"""
        return np.sum((timings - 50) ** 2)
    
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
    
    print(f"{'~' * 5} OPTIMIZATION MODULE UNIT TESTS {'~' * 5}\n")
    
    tests = [
        test_pso_initialization,
        test_pso_optimize
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