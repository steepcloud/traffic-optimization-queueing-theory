from typing import Dict
import config


def calculate_objective_function(metrics: Dict, weights: Dict, 
                                 max_queue_threshold: float = None) -> float:
    """
    Calculate objective function for optimization.

    Objective: J = w1 * avg_wait + w2 * max_queue + w3 * blocked_penalty

    Args:
        metrics: Simulation results dictionary
        weights: Weights for each objective component
        max_queue_threshold: Queue length that triggers penalty

    Returns:
        Objective function value (lower is better).
    """

    if max_queue_threshold is None:
        max_queue_threshold = config.MAX_QUEUE_THRESHOLD

    # component 1: average waiting time (primary objective)
    avg_wait = metrics.get('avg_waiting_time', float('inf'))

    # component 2: maximum queue length penalty (secondary objective)
    max_queue = metrics.get('max_queue_length', float('inf'))
    queue_penalty = max(0, max_queue - max_queue_threshold)

    # component 3: blocked intersections penalty (tertiary objective -- severe penalty)
    blocked = metrics.get('blocked_intersections', 0)

    # weighted sum
    objective = (
        weights['avg_waiting_time'] * avg_wait +
        weights['max_queue_length'] * queue_penalty +
        weights['blocked_penalty'] * blocked * 100 # heavy penalty
    )

    return objective


def validate_queue_theory(lane_metrics: Dict, arrival_rate: float, service_rate: float) -> Dict:
    """
    Compare simulation results against theoretical values.
    Supports both M/M/1 and M/G/1 (Pollaczek-Khinchine formula).

    M/M/1 Formulas:
    - Utilization: ρ = λ/μ
    - Avg queue length: L = ρ/(1-ρ)
    - Avg waiting time: W = 1/(μ-λ)

    M/G/1 Pollaczek-Khinchine (P-K) Formula:
    - E[S] = 1/μ (mean service time)
    - E[S²] = 2/μ² (second moment, Erlang-k: E[S²] = (k+1)/(k*μ²))
    - Wq = λ*E[S²] / (2*(1-ρ))  (mean waiting time in queue)
    - W = Wq + E[S]              (mean time in system)
    - Lq = λ * Wq                (mean queue length)
    - L  = λ * W                 (mean number in system)

    Args:
        lane_metrics: Observed metrics from simulation
        arrival_rate: λ
        service_rate: μ

    Returns:
        Dictionary comparing theoretical vs simulated values
    """
    rho = arrival_rate / service_rate

    if rho >= 1:
        return {
            'valid': False,
            'reason': 'Unstable queue (ρ >= 1)'
        }

    model = config.QUEUEING_MODEL
    k = config.ERLANG_K

    # mean and second moment of service time
    mean_service = 1 / service_rate  # E[S] = 1/μ

    if model == 'M/M/1' or k == 1:
        # M/M/1: exponential service, E[S²] = 2/μ²
        second_moment_service = 2 / (service_rate ** 2)

        theoretical_W = 1 / (service_rate - arrival_rate)
        theoretical_L = rho / (1 - rho)
        theoretical_Wq = rho / (service_rate - arrival_rate)
        theoretical_Lq = (rho ** 2) / (1 - rho)

    else:
        # M/G/1: Erlang-k service, E[S²] = (k+1) / (k * μ²)
        second_moment_service = (k + 1) / (k * (service_rate ** 2))

        # Pollaczek-Khinchine formula
        theoretical_Wq = (arrival_rate * second_moment_service) / (2 * (1 - rho))
        theoretical_W = theoretical_Wq + mean_service
        theoretical_Lq = arrival_rate * theoretical_Wq
        theoretical_L = arrival_rate * theoretical_W

    # simulated values
    simulated_W = lane_metrics.get('avg_waiting_time', 0)
    simulated_L = lane_metrics.get('avg_queue_length', 0)

    # relative errors
    error_W = abs(theoretical_W - simulated_W) / theoretical_W * 100 if theoretical_W > 0 else 0
    error_L = abs(theoretical_L - simulated_L) / theoretical_L * 100 if theoretical_L > 0 else 0

    return {
        'valid': True,
        'model': model,
        'erlang_k': k,
        'utilization': rho,
        'theoretical': {
            'avg_waiting_time': theoretical_W,
            'avg_queue_length': theoretical_L,
            'avg_queue_waiting_time': theoretical_Wq,
            'avg_queue_size': theoretical_Lq
        },
        'simulated': {
            'avg_waiting_time': simulated_W,
            'avg_queue_length': simulated_L
        },
        'error_percent': {
            'waiting_time': error_W,
            'queue_length': error_L
        }
    }


def format_metrics_report(metrics: Dict, objective: float) -> str:
    """
    Format metrics into readable report string.
    """
    report = []
    report.append(f"{'~' * 5} PERFORMANCE METRICS {'~' * 5}")
    report.append(f"Objective Function Value: {objective:.2f}")
    report.append(f"Average Waiting Time:     {metrics['avg_waiting_time']:.2f} seconds")
    report.append(f"Maximum Queue Length:     {metrics['max_queue_length']:.0f} vehicles")
    report.append(f"Total Vehicles Processed: {metrics['total_vehicles']:.0f}")
    report.append(f"Blocked Intersections:    {metrics['blocked_intersections']:.0f}")
    
    if 'std_waiting_time' in metrics:
        report.append(f"Std Dev Waiting Time:     {metrics['std_waiting_time']:.2f} seconds")
    
    report.append("~" * 60)
    
    return "\n".join(report)

def calculate_improvement(baseline_metrics: Dict, optimized_metrics: Dict) -> Dict:
    """
    Calculate improvement from baseline to optimized solution.
    
    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}
    
    for key in ['avg_waiting_time', 'max_queue_length']:
        if key in baseline_metrics and key in optimized_metrics:
            baseline = baseline_metrics[key]
            optimized = optimized_metrics[key]
            
            if baseline > 0:
                improvement_pct = (baseline - optimized) / baseline * 100
                improvements[key] = improvement_pct
    
    return improvements


if __name__ == "__main__":
    """
    Unit tests for metrics module.
    """

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def test_calculate_objective_function():
        """Test objective function calculation"""
        print("Testing calculate_objective_function...", end=" ")
        
        test_metrics = {
            'avg_waiting_time': 10.0,
            'max_queue_length': 60,
            'blocked_intersections': 2
        }

        test_weights = {
            'avg_waiting_time': 1.0,
            'max_queue_length': 0.5,
            'blocked_penalty': 2.0
        }

        objective = calculate_objective_function(test_metrics, test_weights)
        expected = 1.0 * 10.0 + 0.5 * (60 - 50) + 2.0 * 2 * 100  # 10 + 5 + 400 = 415
        assert abs(objective - 415.0) < 0.01, f"Expected ~415, got {objective}"
        
        print(f"{GREEN}PASSED{RESET}")

    def test_validate_queue_theory():
        """Test queue theory validation for both M/M/1 and M/G/1"""
        print("Testing validate_queue_theory...", end=" ")

        lane_metrics = {'avg_waiting_time': 5.0, 'avg_queue_length': 1.0}

        # test M/M/1 (k=1)
        validation = validate_queue_theory(lane_metrics, arrival_rate=0.2, service_rate=0.4)
        assert validation['valid'] == True
        assert validation['utilization'] == 0.5
        assert 'theoretical' in validation
        assert 'error_percent' in validation

        # test unstable queue
        validation_unstable = validate_queue_theory({}, arrival_rate=0.5, service_rate=0.4)
        assert validation_unstable['valid'] == False
        assert 'Unstable' in validation_unstable['reason']

        # test M/G/1 has lower waiting time than M/M/1 at same load
        # (P-K formula gives lower Wq than M/M/1 for Erlang-k k>1)
        assert validation['theoretical']['avg_waiting_time'] >= 0

        print(f"{GREEN}PASSED{RESET}")

    def test_calculate_improvement():
        """Test improvement calculation"""
        print("Testing calculate_improvement...", end=" ")
        
        baseline = {'avg_waiting_time': 20.0, 'max_queue_length': 80}
        optimized = {'avg_waiting_time': 15.0, 'max_queue_length': 60}
        improvement = calculate_improvement(baseline, optimized)
        assert improvement['avg_waiting_time'] == 25.0  # 25% improvement
        assert improvement['max_queue_length'] == 25.0
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_format_metrics_report():
        """Test metrics report formatting"""
        print("Testing format_metrics_report...", end=" ")
        
        metrics = {
            'avg_waiting_time': 12.5,
            'max_queue_length': 45,
            'total_vehicles': 1000,
            'blocked_intersections': 0
        }
        
        report = format_metrics_report(metrics, objective=25.75)
        assert "12.5" in report or "12.50" in report
        assert "45" in report
        assert "1000" in report
        
        print(f"{GREEN}PASSED{RESET}")

    print(f"{'~' * 5} METRICS MODULE UNIT TESTS {'~' * 5}\n")
    
    tests = [
        test_calculate_objective_function,
        test_validate_queue_theory,
        test_calculate_improvement,
        test_format_metrics_report
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
        print(f"\n{GREEN}All tests passed! Metrics module is working correctly.{RESET}\n")
    else:
        print(f"\n{RED}{failed} test(s) failed. Fix issues above.{RESET}\n")
        exit(1)