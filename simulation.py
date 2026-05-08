import simpy
import random as rd
import numpy as np
from typing import Dict, List, Tuple
from network import Network, Intersection, Lane
import config


class Vehicle:
    """Single vehicle travelling through the network."""

    def __init__(self, vehicle_id: int, arrival_time: float, lane: Lane):
        self.vehicle_id = vehicle_id
        self.arrival_time = arrival_time
        self.lane = lane
        self.departure_time = None
        self.waiting_time = 0.0

    def calculate_waiting_time(self, departure_time: float):
        self.departure_time = departure_time
        self.waiting_time = departure_time - self.arrival_time


class TrafficSimulation:
    """
    SimPy-based traffic simulation using M/G/1 queues.

    Each lane is modelled as an M/G/1 queue with Erlang-k service times.
    Inter-arrival times use k=1 (exponential / Poisson arrivals).
    """

    def __init__(self, network: Network, duration: float, warmup: float = 0,
                 random_seed: int = None, verbose: int = 0):
        """
        Args:
            network: Traffic network to simulate.
            duration: Simulation duration (seconds).
            warmup: Warm-up period whose vehicles are excluded from metrics (seconds).
            random_seed: Seed for reproducibility.
            verbose: 0=silent, 1=basic, 2=detailed.
        """
        self.network = network
        self.duration = duration
        self.warmup = warmup
        self.verbose = verbose

        self.env = simpy.Environment()

        if random_seed is not None:
            rd.seed(random_seed)
            np.random.seed(random_seed)

        self.vehicles: List[Vehicle] = []
        self.vehicle_counter = 0

        all_lanes = network.get_all_lanes()

        self.queue_length_samples: Dict[int, List[Tuple[float, int]]] = {
            lane.lane_id: [] for lane in all_lanes
        }
        self.light_state_samples: Dict[int, List[Tuple[float, int]]] = {
            int_id: [] for int_id in network.intersections
        }
        self.lane_servers: Dict[int, simpy.Resource] = {
            lane.lane_id: simpy.Resource(self.env, capacity=1) for lane in all_lanes
        }

    def run(self) -> Dict:
        """Run the simulation and return performance metrics."""
        self.network.reset_all_stats()
        self.vehicles = []
        self.vehicle_counter = 0

        for intersection in self.network.intersections.values():
            self.env.process(self.traffic_light_controller(intersection))
            for lane in intersection.lanes.values():
                self.env.process(self.vehicle_arrivals(intersection, lane))

        self.env.process(self.monitor_queues())
        self.env.run(until=self.duration)

        return self.calculate_metrics()

    def traffic_light_controller(self, intersection: Intersection):
        """SimPy process: cycles the traffic light through its 4 phases."""
        light = intersection.traffic_light

        while True:
            self.light_state_samples[intersection.intersection_id].append(
                (self.env.now, light.current_phase)
            )

            phase_duration = light.get_phase_duration(light.current_phase)

            if self.verbose >= 2:
                print(f"[{self.env.now:.1f}s] Intersection {intersection.intersection_id}: "
                      f"Phase {light.current_phase} for {phase_duration}s")

            yield self.env.timeout(phase_duration)
            light.current_phase = (light.current_phase + 1) % 4

    def vehicle_arrivals(self, intersection: Intersection, lane: Lane):
        """
        SimPy process: generates vehicles with Erlang-k=1 (exponential) inter-arrival times.

        k is fixed at 1 here to produce Poisson arrivals (M in M/G/1).
        Service times use config.ERLANG_K which may differ.
        """
        while True:
            inter_arrival_time = rd.gammavariate(1, 1.0 / lane.arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            vehicle = Vehicle(
                vehicle_id=self.vehicle_counter,
                arrival_time=self.env.now,
                lane=lane
            )
            self.vehicle_counter += 1
            self.vehicles.append(vehicle)

            lane.current_queue_length += 1
            lane.total_vehicles_arrived += 1

            self.env.process(self.vehicle_service(intersection, lane, vehicle))

    def vehicle_service(self, intersection: Intersection, lane: Lane, vehicle: Vehicle):
        """SimPy process: vehicle waits for a server slot and a green light, then departs."""
        server = self.lane_servers[lane.lane_id]
        with server.request() as request:
            yield request  # wait for server to be free (M/G/1 queue discipline)

            while not intersection.traffic_light.is_green(lane.direction):
                yield self.env.timeout(0.1)

            k = config.ERLANG_K
            service_time = rd.gammavariate(k, 1.0 / (lane.service_rate * k))
            yield self.env.timeout(service_time)

            lane.current_queue_length -= 1
            lane.total_vehicles_served += 1
            vehicle.calculate_waiting_time(self.env.now)

            if self.env.now >= self.warmup:
                lane.total_waiting_time += vehicle.waiting_time

    def monitor_queues(self):
        """SimPy process: samples queue lengths every 10 seconds."""
        while True:
            yield self.env.timeout(10)
            for lane in self.network.get_all_lanes():
                server = self.lane_servers[lane.lane_id]
                self.queue_length_samples[lane.lane_id].append(
                    (self.env.now, len(server.queue) + server.count)
                )

    def calculate_metrics(self) -> Dict:
        """Aggregate simulation results into a metrics dictionary."""
        valid_vehicles = [v for v in self.vehicles if v.arrival_time >= self.warmup]

        if not valid_vehicles:
            return {
                'avg_waiting_time': float('inf'),
                'max_queue_length': float('inf'),
                'total_vehicles': 0,
                'blocked_intersections': 0
            }

        avg_waiting_time = np.mean([v.waiting_time for v in valid_vehicles])

        all_lanes = self.network.get_all_lanes()
        lane_maxes = [
            max((q for _, q in self.queue_length_samples[lane.lane_id]), default=0)
            for lane in all_lanes
        ]

        max_queue = max(lane_maxes, default=0)
        blocked_count = sum(1 for m in lane_maxes if m > config.MAX_QUEUE_THRESHOLD)

        lane_avgs = [
            np.mean([q for _, q in self.queue_length_samples[lane.lane_id]])
            for lane in all_lanes
            if self.queue_length_samples[lane.lane_id]
        ]
        avg_queue_length = np.mean(lane_avgs) if lane_avgs else 0.0

        metrics = {
            'avg_waiting_time': avg_waiting_time,
            'max_queue_length': max_queue,
            'total_vehicles': len(valid_vehicles),
            'blocked_intersections': blocked_count,
            'avg_queue_length': avg_queue_length,
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
                             verbose: int = 0) -> Dict:
    """
    Run multiple simulations and return averaged metrics.

    Averaging across runs reduces the effect of stochastic variation.
    queue_samples and light_states are taken from the final run only
    (used downstream for animation).

    Args:
        network: Traffic network to simulate.
        num_runs: Number of independent runs.
        duration: Duration of each run (seconds).
        warmup: Warm-up period to discard (seconds).
        random_seed: Base seed — each run gets seed + run index.
        verbose: 0=silent, 1=print per-run results.

    Returns:
        Averaged metrics dict with queue_samples and light_states from the last run.
    """
    all_metrics = []

    for run in range(num_runs):
        seed = random_seed + run if random_seed is not None else None
        metrics = TrafficSimulation(network, duration, warmup, seed, verbose=0).run()
        all_metrics.append(metrics)

        if verbose >= 1:
            print(f"Run {run+1}/{num_runs}: Avg Wait = {metrics['avg_waiting_time']:.2f}s")

    last = all_metrics[-1]

    return {
        'avg_waiting_time':    np.mean([m['avg_waiting_time'] for m in all_metrics]),
        'max_queue_length':    np.mean([m['max_queue_length'] for m in all_metrics]),
        'total_vehicles':      np.mean([m['total_vehicles'] for m in all_metrics]),
        'blocked_intersections': np.mean([m['blocked_intersections'] for m in all_metrics]),
        'std_waiting_time':    np.std([m['avg_waiting_time'] for m in all_metrics]),
        'queue_samples':       last['queue_samples'],
        'light_states':        last['light_states']
    }


if __name__ == "__main__":
    network = Network(
        num_intersections=4,
        topology=config.NETWORK_TOPOLOGY,
        arrival_rate=config.ARRIVAL_RATE,
        service_rate=config.SERVICE_RATE,
        initial_green=config.INITIAL_GREEN_TIME
    )

    print("Running 100-second test simulation...")
    sim = TrafficSimulation(network, duration=100, warmup=0, random_seed=42, verbose=1)
    metrics = sim.run()

    print("\nSimulation completed successfully!")
    print(f"Vehicles processed: {metrics['total_vehicles']}")
    print(f"Avg waiting time: {metrics['avg_waiting_time']:.2f}s")

    assert metrics['total_vehicles'] > 0, "No vehicles processed!"
    assert metrics['avg_waiting_time'] > 0, "Invalid waiting time!"

    print("\nAll tests passed!")