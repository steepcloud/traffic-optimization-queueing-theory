import simpy
import random as rd
import numpy as np
from typing import Dict, List, Tuple
from network import Network, Intersection, Lane
import config

class Vehicle:
    """
    Represents a single vehicle in the system.
    """

    def __init__(self, vehicle_id: int, arrival_time: float, lane: Lane):
        self.vehicle_id = vehicle_id
        self.arrival_time = arrival_time
        self.lane = lane
        self.departure_time = None
        self.waiting_time = 0.0

    def calculate_waiting_time(self, departure_time: float):
        """Calculate total time spent in system."""
        self.departure_time = departure_time
        self.waiting_time = self.departure_time - self.arrival_time


class TrafficSimulation:
    """
    Main simulation engine using SimPy.
    Implements M/M/1 queueuing for each lane.
    """

    def __init__(self, network: Network, duration: float, warmup: float = 0,
                 random_seed: int = None, verbose: int = 0):
        """
        Args:
            network: Traffic network to simulate
            duration: Simulation duration (seconds)
            warmup: Warm-up period to discard (seconds)
            random_seed: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        """
        self.network = network
        self.duration = duration
        self.warmup = warmup
        self.verbose = verbose

        # simpy env
        self.env = simpy.Environment()

        if random_seed is not None:
            rd.seed(random_seed)
            np.random.seed(random_seed)
        
        # track all vehicles
        self.vehicles: List[Vehicle] = []
        self.vehicle_counter = 0

        # queue length tracking (for time-average calculations)
        self.queue_length_samples: Dict[int, List[Tuple[float, int]]] = {}
        for lane in network.get_all_lanes():
            self.queue_length_samples[lane.lane_id] = []
        
        # traffic light state tracking
        self.light_state_samples: Dict[int, List[Tuple[float, int]]] = {}
        for int_id in network.intersections.keys():
            self.light_state_samples[int_id] = []
        
    def run(self) -> Dict:
        """
        Run the simulation and return performance metrics.
        Returns:
            Dictionary with metrics: avg_waiting_time, max_queue, etc.
        """
        # reset network statistics
        self.network.reset_all_stats()
        self.vehicles = []
        self.vehicle_counter = 0

        # start processes for each intersection
        for intersection in self.network.intersections.values():
            # traffic light controller
            self.env.process(self.traffic_light_controller(intersection))

            # vehicle arrival processes for each incoming lane
            for lane in intersection.lanes.values():
                self.env.process(self.vehicle_arrivals(intersection, lane))

        # start queue monitoring
        self.env.process(self.monitor_queues())

        # run simulation
        self.env.run(until=self.duration)

        # calculate and return metrics
        return self.calculate_metrics()

    def traffic_light_controller(self, intersection: Intersection):
        """
        Process that controls traffic light phase changes.
        """
        light = intersection.traffic_light

        while True:
            # track current phase state
            self.light_state_samples[intersection.intersection_id].append(
                (self.env.now, light.current_phase)
            )

            # get current phase duration
            phase_duration = light.get_phase_duration(light.current_phase)

            if self.verbose >= 2:
                print(f"[{self.env.now:.1f}s] Intersection {intersection.intersection_id}: "
                      f"Phase {light.current_phase} for {phase_duration}s")
            
            # wait for phase duration
            yield self.env.timeout(phase_duration)

            # move to the next phase
            light.current_phase = (light.current_phase + 1) % 4 # assuming 4 phases, todo: maybe modify this

    def vehicle_arrivals(self, intersection: Intersection, lane: Lane):
        """
        Process that generates vehicle arrivals (Poisson process).
        M/M/1: Exponential inter-arrival times.
        """

        while True:
            # exponential inter-arrival time (Poisson process)
            inter_arrival_time = rd.expovariate(lane.arrival_rate)
            yield self.env.timeout(inter_arrival_time)
        
            # create new vehicle
            vehicle = Vehicle(vehicle_id=self.vehicle_counter,
                              arrival_time=self.env.now,
                              lane=lane)
            self.vehicle_counter += 1
            self.vehicles.append(vehicle)

            # add to lane queue
            lane.current_queue_length += 1
            lane.total_vehicles_arrived += 1

            # start vehicle service process
            self.env.process(self.vehicle_service(intersection, lane, vehicle))

    # todo: further m/g/1 implementation
    '''
    def vehicle_arrivals(self, intersection: Intersection, lane: Lane):
        while True:
        # Erlang-2 inter-arrival time (k=2 shape parameter)
        # this models cars arriving in small groups (platooning)
        k = 2 # shape paramter (higher = more regular arrivals)
        theta = k / lane.arrival_rate # scale parameter
        inter_arrival_time = rd.gammavariate(k, theta)
        yield self.env.timeout(inter_arrival_time)
        # etc etc we have the code above
    '''

    def vehicle_service(self, intersection: Intersection, lane: Lane, vehicle: Vehicle):
        """
        Process for a vehicle waiting in queue and departing.
        """
        # wait until green light
        while not intersection.traffic_light.is_green(lane.direction):
            yield self.env.timeout(0.1) # check every 0.1s
        
        # service time (exponential for M/M/1)
        service_time = rd.expovariate(lane.service_rate)
        yield self.env.timeout(service_time)
        
        # vehicle departs
        lane.current_queue_length -= 1
        lane.total_vehicles_served += 1

        vehicle.calculate_waiting_time(self.env.now)

        # only count vehicles after warm-up period
        if self.env.now >= self.warmup:
            lane.total_waiting_time += vehicle.waiting_time
    
    def monitor_queues(self):
        """
        Periodically sample queue lengths for time-average calculations.
        """
        while True:
            yield self.env.timeout(10) # sample every 10 seconds

            # record queue lengths
            for lane in self.network.get_all_lanes():
                self.queue_length_samples[lane.lane_id].append(
                    (self.env.now, lane.current_queue_length)
                )

    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from simulation results.
        """
        # filter vehicles that arrived after warm-up
        valid_vehicles = [v for v in self.vehicles if v.arrival_time >= self.warmup]

        if len(valid_vehicles) == 0:
            return {
                'avg_waiting_time': float('inf'),
                'max_queue_length': float('inf'),
                'total_vehicles': 0,
                'blocked_intersections': 0
            }
        
        # average waiting time
        avg_waiting_time = np.mean([v.waiting_time for v in valid_vehicles])

        # max queue length across all lanes
        max_queue = 0
        for lane in self.network.get_all_lanes():
            lane_max = max([q for t, q in self.queue_length_samples[lane.lane_id]], default=0)
            max_queue = max(max_queue, lane_max)
        
        # count blocked intersections (queue > threshold)
        blocked_count = 0
        for lane in self.network.get_all_lanes():
            if lane.current_queue_length > config.MAX_QUEUE_THRESHOLD:
                blocked_count += 1

        metrics = {
            'avg_waiting_time': avg_waiting_time,
            'max_queue_length': max_queue,
            'total_vehicles': len(valid_vehicles),
            'blocked_intersections': blocked_count,
            'avg_queue_length': np.mean([lane.current_queue_length
                                         for lane in self.network.get_all_lanes()]),
            'queue_samples': self.queue_length_samples,
            'light_states': self.light_state_samples
        }

        if self.verbose >= 1:
            print(f"\n~~ Simulation Results ~~")
            print(f"Total vehicles processed: {metrics['total_vehicles']}")
            print(f"Average waiting time: {metrics['avg_waiting_time']:.2f}s")
            print(f"Max queue length: {metrics['max_queue_length']}")
            print(f"Blocked lanes: {metrics['blocked_intersections']}")
        
        return metrics


def run_multiple_simulations(network: Network, num_runs: int, duration: float,
                             warmup: float = 0, random_seed: int = None,
                             verbose: int = 0) -> List[Dict]:
    """
    Run multiple simulations and average the results (handle stochasticity).

    Args:
        network: Traffic network to simulate
        num_runs: Number of simulation runs
        duration: Duration of each simulation (seconds)
        warmup: Warm-up period to discard (seconds)
        random_seed: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
    
    Returns:
        Average metrics across all runs.
    """
    all_metrics = []

    for run in range(num_runs):
        seed = random_seed + run if random_seed is not None else None
        
        sim = TrafficSimulation(network, duration, warmup, seed, verbose=0)
        metrics = sim.run()
        all_metrics.append(metrics)
        
        if verbose >= 1:
            print(f"Run {run+1}/{num_runs}: Avg Wait = {metrics['avg_waiting_time']:.2f}s")
    
    # average across all runs
    averaged = {
        'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in all_metrics]),
        'max_queue_length': np.mean([m['max_queue_length'] for m in all_metrics]),
        'total_vehicles': np.mean([m['total_vehicles'] for m in all_metrics]),
        'blocked_intersections': np.mean([m['blocked_intersections'] for m in all_metrics]),
        'std_waiting_time': np.std([m['avg_waiting_time'] for m in all_metrics])
    }
    
    return averaged
    

if __name__ == "__main__":
    """
    Basic test of simulation module.
    """
    from network import Network

    network = Network(
        num_intersections=4,
        topology=config.NETWORK_TOPOLOGY,
        arrival_rate=config.ARRIVAL_RATE,
        service_rate=config.SERVICE_RATE,
        initial_green=config.INITIAL_GREEN_TIME
    )

    # run short simulation
    print("Running 100-second test simulation...")
    sim = TrafficSimulation(network, duration=100, warmup=0, random_seed=42, verbose=1)
    metrics = sim.run()
    
    print("\nSimulation completed successfully!")
    print(f"Vehicles processed: {metrics['total_vehicles']}")
    print(f"Avg waiting time: {metrics['avg_waiting_time']:.2f}s")
    
    # sanity checks
    assert metrics['total_vehicles'] > 0, "No vehicles processed!"
    assert metrics['avg_waiting_time'] > 0, "Invalid waiting time!"
    
    print("\nAll tests passed!")