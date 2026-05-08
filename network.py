import numpy as np
from typing import List, Dict, Tuple
import config

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


class Lane:
    """Single lane (queue) at an intersection."""

    def __init__(self, lane_id: int, direction: str, arrival_rate: float, service_rate: float):
        """
        Args:
            lane_id: Unique identifier for the lane.
            direction: One of 'N', 'S', 'E', 'W'.
            arrival_rate: λ in vehicles/second.
            service_rate: μ in vehicles/second.
        """
        if arrival_rate <= 0:
            raise ValueError(f"Arrival rate must be positive, got {arrival_rate}")
        if service_rate <= 0:
            raise ValueError(f"Service rate must be positive, got {service_rate}")
        if direction not in ('N', 'S', 'E', 'W'):
            raise ValueError(f"Direction must be N/S/E/W, got {direction}")

        self.lane_id = lane_id
        self.direction = direction
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

        self.current_queue_length = 0
        self.total_vehicles_served = 0
        self.total_vehicles_arrived = 0
        self.total_waiting_time = 0.0

        self.utilization = arrival_rate / service_rate

        if self.utilization < 1:
            self.theoretical_avg_queue_length = self.utilization / (1 - self.utilization)
            self.theoretical_avg_waiting_time = 1 / (service_rate - arrival_rate)
        else:
            self.theoretical_avg_queue_length = float('inf')
            self.theoretical_avg_waiting_time = float('inf')

    def reset_stats(self):
        """Reset statistics for a new simulation run."""
        self.current_queue_length = 0
        self.total_vehicles_served = 0
        self.total_vehicles_arrived = 0
        self.total_waiting_time = 0.0

    def __repr__(self):
        return f"Lane({self.direction}, λ={self.arrival_rate:.2f}, μ={self.service_rate:.2f}, ρ={self.utilization:.2f})"


class TrafficLight:
    """
    4-phase signal controller: NS green → NS yellow → EW green → EW yellow.

    Phase indices:
        0 = NS green, 1 = NS yellow, 2 = EW green, 3 = EW yellow
    """

    _PHASE_DURATIONS = None  # populated in __init__ since it depends on instance state

    def __init__(self, green_ns: float, green_ew: float, yellow: float = 3.0, always_green: bool = False):
        """
        Args:
            green_ns: Green duration for North-South lanes (seconds).
            green_ew: Green duration for East-West lanes (seconds).
            yellow: Yellow transition duration (seconds).
            always_green: If True, all directions are always green (used for M/G/1 validation).
        """
        self.green_ns = green_ns
        self.green_ew = green_ew
        self.yellow = yellow
        self.always_green = always_green
        self.cycle_time = green_ns + green_ew + 2 * yellow
        self.current_phase = 0
        self.time_in_phase = 0.0

    def update_timings(self, green_ns: float, green_ew: float):
        """Update green durations and recalculate cycle time (called by optimizer)."""
        self.green_ns = green_ns
        self.green_ew = green_ew
        self.cycle_time = green_ns + green_ew + 2 * self.yellow

    def get_phase_duration(self, phase: int) -> float:
        """Return the duration of the given phase index."""
        durations = [self.green_ns, self.yellow, self.green_ew, self.yellow]
        if not 0 <= phase < len(durations):
            raise ValueError(f"Invalid phase: {phase}")
        return durations[phase]

    def is_green(self, direction: str) -> bool:
        """Return True if the signal is green for the given direction."""
        if self.always_green:
            return True
        if self.current_phase == 0:
            return direction in ('N', 'S')
        if self.current_phase == 2:
            return direction in ('E', 'W')
        return False

    def __repr__(self):
        return f"TrafficLight(NS={self.green_ns}s, EW={self.green_ew}s, Cycle={self.cycle_time}s)"


class Intersection:
    """Traffic intersection with four directional lanes and a traffic light."""

    def __init__(self, intersection_id: int, arrival_rate: float, service_rate: float,
                 green_ns: float, green_ew: float, always_green: bool = False):
        """
        Args:
            intersection_id: Unique identifier.
            arrival_rate: λ for each lane (vehicles/second).
            service_rate: μ for each lane (vehicles/second).
            green_ns: Initial NS green duration (seconds).
            green_ew: Initial EW green duration (seconds).
            always_green: If True, all directions are always green (for validation).
        """
        self.intersection_id = intersection_id
        self.lanes: Dict[str, Lane] = {
            direction: Lane(intersection_id * 4 + i, direction, arrival_rate, service_rate)
            for i, direction in enumerate(('N', 'S', 'E', 'W'))
        }
        self.traffic_light = TrafficLight(green_ns=green_ns, green_ew=green_ew, always_green=always_green)
        self.outgoing_connections: Dict[str, int] = {}

    def reset_stats(self):
        """Reset statistics for all lanes."""
        for lane in self.lanes.values():
            lane.reset_stats()

    def __repr__(self):
        return f"Intersection({self.intersection_id}, {len(self.lanes)} lanes)"


class Network:
    """Complete traffic network: multiple intersections connected together."""

    # Known spatial layouts keyed by (num_intersections, is_linear_4)
    _KNOWN_POSITIONS = {
        1: {0: (0, 0)},
        9: {
            0: (0, 2), 1: (1, 2), 2: (2, 2),
            3: (0, 1), 4: (1, 1), 5: (2, 1),
            6: (0, 0), 7: (1, 0), 8: (2, 0)
        }
    }

    _LINEAR_4_TOPOLOGY = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    _LINEAR_4_POSITIONS = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    _GRID_4_POSITIONS   = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

    def __init__(self, num_intersections: int, topology: Dict[int, List[int]],
                 arrival_rate: float, service_rate: float, initial_green: float,
                 always_green: bool = False):
        """
        Args:
            num_intersections: Total number of intersections.
            topology: Graph structure {intersection_id: [connected_ids]}.
            arrival_rate: Default λ for all lanes (vehicles/second).
            service_rate: Default μ for all lanes (vehicles/second).
            initial_green: Initial green duration for all intersections (seconds).
            always_green: If True, all directions are always green (for validation).
        """
        self.num_intersections = num_intersections
        self.topology = topology
        self.intersections: Dict[int, Intersection] = {}

        for i in range(num_intersections):
            if getattr(config, 'USE_ASYMMETRIC_TRAFFIC', False):
                self.intersections[i] = self._create_asymmetric_intersection(
                    i, service_rate, initial_green, always_green
                )
            else:
                self.intersections[i] = Intersection(
                    intersection_id=i,
                    arrival_rate=arrival_rate,
                    service_rate=service_rate,
                    green_ns=initial_green,
                    green_ew=initial_green,
                    always_green=always_green
                )

        self._setup_connections()

    def _resolve_positions(self) -> Dict[int, Tuple[int, int]]:
        """
        Return spatial (x, y) positions for all intersections.

        Checks for manually set positions first, then falls back to known
        layouts for n=1, n=4 (grid and linear), n=9, and finally a linear
        fallback for arbitrary sizes.
        """
        if hasattr(self, 'positions'):
            return self.positions
        if self.num_intersections in self._KNOWN_POSITIONS:
            return self._KNOWN_POSITIONS[self.num_intersections]
        if self.num_intersections == 4:
            if self.topology == self._LINEAR_4_TOPOLOGY:
                return self._LINEAR_4_POSITIONS
            return self._GRID_4_POSITIONS
        return {i: (i, 0) for i in range(self.num_intersections)}

    def _setup_connections(self):
        """
        Assign outgoing_connections on each intersection based on spatial layout.

        For a 2×2 grid (default config):
            [0]----[1]
            |      |
            [2]----[3]

        dx > 0 → East, dx < 0 → West, dy > 0 → North, dy < 0 → South.
        Diagonal neighbours are skipped.
        """
        positions = self._resolve_positions()

        for int_id, connected_ids in self.topology.items():
            if int_id not in self.intersections:
                continue

            current_pos = positions[int_id]
            connections = {}

            for connected_id in connected_ids:
                if connected_id not in positions:
                    continue
                dx = positions[connected_id][0] - current_pos[0]
                dy = positions[connected_id][1] - current_pos[1]

                if   dx > 0 and dy == 0: connections['E'] = connected_id
                elif dx < 0 and dy == 0: connections['W'] = connected_id
                elif dx == 0 and dy > 0: connections['N'] = connected_id
                elif dx == 0 and dy < 0: connections['S'] = connected_id
                # diagonal neighbours are silently skipped

            self.intersections[int_id].outgoing_connections = connections

    def _create_asymmetric_intersection(self, int_id: int, service_rate: float,
                                        initial_green: float, always_green: bool = False) -> Intersection:
        """Create an intersection with per-lane arrival rates from config.LANE_ARRIVAL_RATES."""
        intersection = Intersection.__new__(Intersection)
        intersection.intersection_id = int_id
        intersection.outgoing_connections = {}
        intersection.lanes = {
            direction: Lane(
                int_id * 4 + i,
                direction,
                config.LANE_ARRIVAL_RATES[int_id][direction],
                service_rate
            )
            for i, direction in enumerate(('N', 'S', 'E', 'W'))
        }
        intersection.traffic_light = TrafficLight(
            green_ns=initial_green, green_ew=initial_green, always_green=always_green
        )
        return intersection

    def set_custom_positions(self, positions: Dict[int, Tuple[int, int]]):
        """
        Override spatial positions and rebuild connections.

        Args:
            positions: {intersection_id: (x, y)} coordinates.

        Example:
            network.set_custom_positions({0: (0, 0), 1: (1, 0), 2: (2, 0)})
        """
        self.positions = positions
        self._setup_connections()

    def reset_all_stats(self):
        """Reset statistics for every intersection in the network."""
        for intersection in self.intersections.values():
            intersection.reset_stats()

    def get_all_lanes(self) -> List[Lane]:
        """Return a flat list of all lanes across all intersections."""
        return [lane for i in self.intersections.values() for lane in i.lanes.values()]

    def update_signal_timings(self, timings: np.ndarray):
        """
        Apply optimizer-produced green times to all traffic lights.

        Args:
            timings: [int0_NS, int0_EW, int1_NS, int1_EW, ...], length = 2 * num_intersections.

        Raises:
            ValueError: If timings length does not match expected size.
        """
        expected = self.num_intersections * 2
        if len(timings) != expected:
            raise ValueError(f"Timings array must have length {expected}, got {len(timings)}")

        for i, intersection in self.intersections.items():
            intersection.traffic_light.update_timings(timings[i * 2], timings[i * 2 + 1])

    def get_network_metrics(self) -> Dict:
        """Return aggregate performance metrics across the entire network."""
        all_lanes = self.get_all_lanes()
        total_waiting_time = sum(lane.total_waiting_time for lane in all_lanes)
        total_vehicles = sum(lane.total_vehicles_served for lane in all_lanes)

        return {
            'avg_waiting_time': total_waiting_time / total_vehicles if total_vehicles > 0 else 0,
            'max_queue_length': max((lane.current_queue_length for lane in all_lanes), default=0),
            'avg_queue_length': np.mean([lane.current_queue_length for lane in all_lanes]),
            'total_vehicles_served': total_vehicles,
            'num_lanes': len(all_lanes)
        }

    def get_positions(self) -> Dict[int, Tuple[int, int]]:
        """Return spatial positions for all intersections."""
        return self._resolve_positions()

    def __repr__(self):
        return f"Network({self.num_intersections} intersections, {len(self.get_all_lanes())} total lanes)"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lane_creation():
    print("Testing Lane creation...", end=" ")

    lane = Lane(0, 'N', 0.2, 0.4)
    assert lane.direction == 'N'
    assert lane.utilization == 0.5
    assert lane.theoretical_avg_waiting_time == 5.0

    try:
        Lane(0, 'X', 0.2, 0.4)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        Lane(0, 'N', -0.2, 0.4)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print(f"{GREEN}PASSED{RESET}")


def test_traffic_light():
    print("Testing TrafficLight...", end=" ")

    light = TrafficLight(green_ns=30, green_ew=40)
    assert light.green_ns == 30
    assert light.green_ew == 40
    assert light.cycle_time == 30 + 40 + 2 * 3
    assert light.is_green('N') == True
    assert light.is_green('E') == False

    light.update_timings(50, 35)
    assert light.green_ns == 50
    assert light.cycle_time == 50 + 35 + 6

    print(f"{GREEN}PASSED{RESET}")


def test_intersection():
    print("Testing Intersection...", end=" ")

    intersection = Intersection(0, 0.2, 0.4, 30, 30)
    assert intersection.intersection_id == 0
    assert len(intersection.lanes) == 4
    assert 'N' in intersection.lanes
    assert intersection.lanes['N'].direction == 'N'

    print(f"{GREEN}PASSED{RESET}")


def test_2x2_grid_connections():
    print("Testing 2×2 grid connections...", end=" ")

    network = Network(
        num_intersections=4,
        topology={0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]},
        arrival_rate=0.2,
        service_rate=0.4,
        initial_green=30
    )

    assert network.intersections[0].outgoing_connections == {'E': 1, 'S': 2}
    assert network.intersections[1].outgoing_connections == {'W': 0, 'S': 3}
    assert network.intersections[2].outgoing_connections == {'N': 0, 'E': 3}
    assert network.intersections[3].outgoing_connections == {'N': 1, 'W': 2}

    print(f"{GREEN}PASSED{RESET}")


def test_linear_corridor_connections():
    print("Testing linear corridor connections...", end=" ")

    network = Network(
        num_intersections=4,
        topology={0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        arrival_rate=0.2,
        service_rate=0.4,
        initial_green=30
    )

    assert network.intersections[0].outgoing_connections == {'E': 1}
    assert network.intersections[1].outgoing_connections == {'W': 0, 'E': 2}
    assert network.intersections[2].outgoing_connections == {'W': 1, 'E': 3}
    assert network.intersections[3].outgoing_connections == {'W': 2}

    print(f"{GREEN}PASSED{RESET}")


def test_3x3_grid_connections():
    print("Testing 3×3 grid connections...", end=" ")

    network = Network(
        num_intersections=9,
        topology={
            0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
            3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
            6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
        },
        arrival_rate=0.2,
        service_rate=0.4,
        initial_green=30
    )

    assert network.intersections[4].outgoing_connections == {'N': 1, 'W': 3, 'E': 5, 'S': 7}
    assert network.intersections[0].outgoing_connections == {'E': 1, 'S': 3}

    print(f"{GREEN}PASSED{RESET}")


def test_network_methods():
    print("Testing Network methods...", end=" ")

    network = Network(4, config.NETWORK_TOPOLOGY, 0.2, 0.4, 30)

    assert len(network.get_all_lanes()) == 16

    new_timings = np.array([35, 40, 25, 50, 30, 45, 40, 35])
    network.update_signal_timings(new_timings)
    assert network.intersections[0].traffic_light.green_ns == 35
    assert network.intersections[0].traffic_light.green_ew == 40
    assert network.intersections[1].traffic_light.green_ns == 25

    try:
        network.update_signal_timings(np.array([30, 30]))
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Timings array must have length 8" in str(e)

    metrics = network.get_network_metrics()
    assert 'avg_waiting_time' in metrics
    assert 'max_queue_length' in metrics
    assert metrics['num_lanes'] == 16

    print(f"{GREEN}PASSED{RESET}")


def test_custom_positions():
    print("Testing custom positions...", end=" ")

    network = Network(3, {0: [1], 1: [0, 2], 2: [1]}, 0.2, 0.4, 30)

    network.set_custom_positions({0: (0, 0), 1: (1, 1), 2: (2, 2)})
    assert len(network.intersections[0].outgoing_connections) == 0

    network.set_custom_positions({0: (0, 0), 1: (1, 0), 2: (2, 0)})
    assert network.intersections[0].outgoing_connections == {'E': 1}
    assert network.intersections[1].outgoing_connections == {'W': 0, 'E': 2}

    print(f"{GREEN}PASSED{RESET}")


def test_mm1_theory_validation():
    print("Testing M/M/1 theory...", end=" ")

    lane = Lane(0, 'N', 0.2, 0.4)
    assert abs(lane.utilization - 0.5) < 0.001
    assert abs(lane.theoretical_avg_queue_length - 1.0) < 0.001
    assert abs(lane.theoretical_avg_waiting_time - 5.0) < 0.001

    unstable = Lane(1, 'S', 0.5, 0.4)
    assert unstable.theoretical_avg_queue_length == float('inf')
    assert unstable.theoretical_avg_waiting_time == float('inf')

    print(f"{GREEN}PASSED{RESET}")


if __name__ == "__main__":
    print("=" * 60)
    print("NETWORK MODULE UNIT TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_lane_creation,
        test_traffic_light,
        test_intersection,
        test_2x2_grid_connections,
        test_linear_corridor_connections,
        test_3x3_grid_connections,
        test_network_methods,
        test_custom_positions,
        test_mm1_theory_validation,
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

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"{GREEN}RESULTS: {passed} passed, {failed} failed{RESET}")
    else:
        print(f"RESULTS: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")
    print("=" * 60)

    if failed == 0:
        print(f"\n{GREEN}All tests passed! Network module is working correctly.{RESET}\n")
    else:
        print(f"\n{RED}{failed} test(s) failed. Fix issues above.{RESET}\n")
        exit(1)