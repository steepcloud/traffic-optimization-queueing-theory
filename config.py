NUM_INTERSECTIONS = 4 # 4-intersection grid
LANES_PER_INTERSECTION = 4 # North, South, East, West

# Network topology - which intersections connect to which
# Format: {intersection_id: [connected_intersection_ids]}
NETWORK_TOPOLOGY = {
    0: [1, 2], # Intersection 0 connects to 1 (East) and 2 (South)
    1: [0, 3], # Intersection 1 connects to 0 (West) and 3 (South)
    2: [0, 3], # Intersection 2 connects to 0 (North) and 3 (East)
    3: [1, 2]  # Intersection 3 connects to 1 (North) and 2 (West)
}

# --- Traffic parameters (M/M/1 queue)
# Arrival rates (λ) - vehicles per second for each lane
# Lower values = less traffic, higher = more congestion
ARRIVAL_RATE = 0.2 # vehicles/second (720 vehicles/hour per lane)

# Service rates (μ) - vehicles per second that can pass through green light
# Dependent on green light duration and intersection capacity / saturation flow rate
SERVICE_RATE = 0.4 # vehicles/second (1440 vehicles/hour per lane) (must be > ARRIVAL_RATE for stability)

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
MIN_GREEN_TIME = 30  # minimum green time (safety constraint)
MAX_GREEN_TIME = 90  # maximum green time (to prevent excessive wait times)

# --- Simulation parameters

# How long to run each simulation (in seconds)
SIMULATION_DURATION = 3600  # 1 hour of simulated traffic

# Warmup period (seconds) - to let traffic stabilize before measurements
WARMUP_PERIOD = 600  # 10 minutes

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of simulation runs for averaging results (handles stochastic variation)
NUM_SIMULATION_RUNS = 10

# --- Optimization parameters (PSO)

PSO_CONFIG = {
    'num_particles': 20,        # Swarm size
    'num_iterations': 50,       # How many generations
    'w': 0.7,                   # Inertia weight (momentum)
    'c1': 1.5,                  # Cognitive coefficient (personal best)
    'c2': 1.5,                  # Social coefficient (global best)
    'bounds': (MIN_GREEN_TIME, MAX_GREEN_TIME)  # Search space
}

# --- Optimization parameters (ACO)

ACO_CONFIG = {
    'num_ants': 20,
    'num_iterations': 50,
    'alpha': 1.0,               # Pheromone importance
    'beta': 2.0,                # Heuristic importance
    'evaporation_rate': 0.1,
    'pheromone_constant': 100
}

# --- Objective function weights

# Multi-objective weights for: J = w1*W + w2*max(Q) + w3*B
OBJECTIVE_WEIGHTS = {
    'avg_waiting_time': 1.0,    # Primary: minimize average delay
    'max_queue_length': 0.5,    # Secondary: prevent overflow
    'blocked_penalty': 2.0      # Tertiary: heavily penalize gridlock
}

# Maximum acceptable queue length (triggers penalty)
MAX_QUEUE_THRESHOLD = 50  # vehicles

# --- Visualization parameters

# Whether to show plots during optimization (slows down, but informative)
SHOW_PLOTS_DURING_OPT = False

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