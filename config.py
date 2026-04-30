NUM_INTERSECTIONS = 4 # 4-intersection grid
LANES_PER_INTERSECTION = 4 # North, South, East, West

# Network topology - which intersections connect to which
# Format: {intersection_id: [connected_intersection_ids]}
NETWORK_TOPOLOGY = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2],
}

# --- Traffic parameters (M/G/1 queue)
# Arrival rates (λ) - vehicles per second for each lane
# Lower values = less traffic, higher = more congestion
ARRIVAL_RATE = 0.05 # vehicles/second (720 vehicles/hour per lane)
# Queueing model
QUEUEING_MODEL = 'M/G/1'    # M/G/1
ERLANG_K = 2                 # shape parameter for Erlang-k distribution
                             # k=1 -> exponential (M/M/1)
                             # k=2 -> Erlang-2 (M/G/1, moderate platooning)
                             # k=5 -> nearly deterministic arrivals

# Asymmetric traffic (realistic scenario) - COMMENT OUT to use ARRIVAL_RATE
LANE_ARRIVAL_RATES = {
    0: {'N': 0.35, 'S': 0.38, 'E': 0.39, 'W': 0.36},  # Higher load
    1: {'N': 0.38, 'S': 0.35, 'E': 0.36, 'W': 0.39},  
    2: {'N': 0.39, 'S': 0.36, 'E': 0.35, 'W': 0.38},  
    3: {'N': 0.36, 'S': 0.39, 'E': 0.38, 'W': 0.35}   # ρ ≈ 0.95
}

# Use asymmetric rates if defined, otherwise uniform
USE_ASYMMETRIC_TRAFFIC = False  # set to False to use uniform ARRIVAL_RATE
# Service rates (μ) - vehicles per second that can pass through green light
# Dependent on green light duration and intersection capacity / saturation flow rate
SERVICE_RATE = 0.15 # vehicles/second (1440 vehicles/hour per lane) (must be > ARRIVAL_RATE for stability)
# Utilization ρ = λ / μ should be < 1 for queue stability
# With current values: ρ = 0.2/0.4 = 0.5 (50% utilization - stable)

# --- Traffic light parameters

# All values in seconds
INITIAL_GREEN_TIME = 30     # green light duration per phase
INITIAL_RED_TIME = 5        # red light duration per phase
YELLOW_TIME = 3             # yellow light transition (fixed, not optimized)

# Total cycle time = GREEN + RED + YELLOW
INITIAL_CYCLE_TIME = INITIAL_GREEN_TIME + INITIAL_RED_TIME + YELLOW_TIME

# Optimization bounds for green light duration
MIN_GREEN_TIME = 20  # minimum green time (safety constraint)
MAX_GREEN_TIME = 90  # maximum green time (to prevent excessive wait times)
# --- Simulation parameters

# How long to run each simulation (in seconds)
SIMULATION_DURATION = 3600  # 30 minutes of simulated traffic
# Warmup period (seconds) - to let traffic stabilize before measurements
WARMUP_PERIOD = 300  # 5 minutes
# Random seed for reproducibility
RANDOM_SEED = 42

# Number of simulation runs for averaging results (handles stochastic variation)
NUM_SIMULATION_RUNS = 5 # (3600 SD, NSR 10 previous)

# --- Optimization parameters (PSO)

PSO_CONFIG = {
    'num_particles': 10,        # Swarm size
    'num_iterations': 20,       # How many generations
    'w': 0.7,                   # Inertia weight (momentum)
    'c1': 1.5,                  # Cognitive coefficient (personal best)
    'c2': 1.5,                  # Social coefficient (global best)
    'bounds': (20, 90)  # Search space
}

# --- Optimization parameters (ACO)

ACO_CONFIG = {
    'n_ants': 10,                                   # Same as PSO num_particles (fair comparison)
    'archive_size': 10,                             # k - solution memory pool
    'q': 0.5,                                       # Locality (small = exploit best, large = explore)
    'xi': 0.85,                                     # Evaporation rate (noise shrink factor)
    'num_iterations': 20,                           # Same as PSO (fair comparison)
    'bounds': (20, 90)      # Search space
}

# --- Objective function weights

# Multi-objective weights for: J = w1*W + w2*max(Q) + w3*B
OBJECTIVE_WEIGHTS = {
    'avg_waiting_time': 1.0,    # Primary: minimize average delay
    'max_queue_length': 0.5,    # Secondary: prevent overflow
    'blocked_penalty': 2.0      # Tertiary: heavily penalize gridlock
}

# Maximum acceptable queue length (triggers penalty)
MAX_QUEUE_THRESHOLD = 20  # vehicles
# --- Visualization parameters

# Whether to show plots during optimization (slows down, but informative)
SHOW_PLOTS_DURING_OPT = True

# Whether to save plots to files
SAVE_PLOTS = True

# Output directory for plots and results
OUTPUT_DIR = "results"

# Animation speed (milliseconds per frame)
ANIMATION_INTERVAL = 100

# --- Debug/logging

# Verbosity level: 0=silent, 1=basic, 2=detailed
VERBOSE = 1

# Print queue status every N seconds during simulation
PRINT_INTERVAL = 300  # Print every 5 minutes (simulated time)