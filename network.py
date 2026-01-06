import numpy as np
from typing import List, Dict, Tuple
import config


class Lane:
    """
    Represents a single lane (queue) at an intersection.
    M/M/1 queue: Poisson arrivals, Exponential service times, 1 server.
    """

    def __init__(self, lane_id: int, direction: str, arrival_rate: float, service_rate: float):
        """
        Args:
            lane_id (int): Unique identifier for the lane.
            direction (str): Direction of the lane ('N', 'S', 'E', 'W').
            arrival_rate (float): Arrival rate (λ) in vehicles/second.
            service_rate (float): Service rate (μ) in vehicles/second.
        """
        
        # validation
        if arrival_rate <= 0:
            raise ValueError(f"Arrival rate must be positive, got {arrival_rate}")
        if service_rate <= 0:
            raise ValueError(f"Service rate must be positive, got {service_rate}")
        if direction not in ['N', 'S', 'E', 'W']:
            raise ValueError(f"Direction must be N/S/E/W, got {direction}")
        
        self.lane_id = lane_id
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.arrival_rate = arrival_rate  # λ
        self.service_rate = service_rate  # μ

        # queue state (updated during simulation)
        self.current_queue_length = 0
        self.total_vehicles_served = 0
        self.total_vehicles_arrived = 0
        self.total_waiting_time = 0.0

        # utilization ρ = λ / μ (should be < 1 for stability)
        self.utilization = arrival_rate / service_rate

        # theoretical m/m/1 metrics (for validation)
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
    Controls which lanes can pass through intersection.
    Simple 2-phase system: North-South green, then East-West green.
    """

    def __init__(self, green_ns: float, green_ew: float, yellow: float = 3.0):
        """
        Args:
            green_ns (float): Green light duration for North-South lanes (seconds).
            green_ew (float): Green light duration for East-West lanes (seconds).
            yellow (float): Yellow light duration (seconds).
        """
        self.green_ns = green_ns
        self.green_ew = green_ew
        self.yellow = yellow

        # total cycle time
        self.cycle_time = green_ns + green_ew + 2 * yellow

        # current phase: 0=NS_green, 1=NS_yellow, 2=EW_green, 3=EW_yellow
        self.current_phase = 0
        self.time_in_phase = 0.0
    
    def update_timings(self, green_ns: float, green_ew: float):
        """Update green light durations and recalculate cycle time (called by optimizer)."""
        self.green_ns = green_ns
        self.green_ew = green_ew
        self.cycle_time = green_ns + green_ew + 2 * self.yellow
    
    def get_phase_duration(self, phase: int) -> float:
        """Get duration of a specific phase."""
        if phase == 0:
            return self.green_ns
        elif phase == 1:
            return self.yellow
        elif phase == 2:
            return self.green_ew
        elif phase == 3:
            return self.yellow
        else:
            raise ValueError("Invalid phase")
    
    def is_green(self, direction: str) -> bool:
        """Check if the light is green for a given direction."""
        if self.current_phase == 0 and direction in ['N', 'S']:
            return True
        elif self.current_phase == 2 and direction in ['E', 'W']:
            return True
        return False

    def __repr__(self):
        return f"TrafficLight(NS={self.green_ns}s, EW={self.green_ew}s, Cycle={self.cycle_time}s)"
    

class Intersection:
    """
    Represents a traffic intersection with multiple lanes and a traffic light.
    """

    def __init__(self, intersection_id: int, arrival_rate: float, service_rate: float,
                 green_ns: float, green_ew: float):
        """
        Args:
            intersection_id (int): Unique identifier for the intersection.
            arrival_rate (float): Arrival rate (λ) for each lane (vehicles/second).
            service_rate (float): Service rate (μ) for each lane (vehicles/second).
            green_ns (float): Initial green light duration for North-South lanes (seconds).
            green_ew (float): Initial green light duration for East-West lanes (seconds).
        """
        self.intersection_id = intersection_id

        # 4 lanes (one per direction)
        self.lanes: Dict[str, Lane] = {}
        for i, direction in enumerate(['N', 'S', 'E', 'W']):
            lane_id = intersection_id * 4 + i
            self.lanes[direction] = Lane(lane_id, direction, arrival_rate, service_rate)

        # traffic light controller
        self.traffic_light = TrafficLight(green_ns=green_ns, green_ew=green_ew)

        # connections to other intersections (set by network)
        self.outgoing_connections: Dict[str, int] = {} # {direction: intersection_id}

    def reset_stats(self):
        """Reset statistics for all lanes."""
        for lane in self.lanes.values():
            lane.reset_stats()
    
    def __repr__(self):
        return f"Intersection({self.intersection_id}, {len(self.lanes)} lanes)"
    

class Network:
    """
    Complete traffic network: multiple intersections connected together.
    """

    def __init__(self, num_intersections: int, topology: Dict[int, List[int]],
                 arrival_rate: float, service_rate: float, initial_green: float):
        """
        Args:
            num_intersections (int): Total number of intersections in the network.
            topology (Dict[int, List[int]]): Graph structure {intersection_id: [connected_ids]}.
            arrival_rate (float): Default arrival rate (λ) for all lanes (vehicles/second).
            service_rate (float): Default service rate (μ) for all lanes (vehicles/second).
            initial_green (float): Initial green light duration for all intersections (seconds).
        """
        self.num_intersections = num_intersections
        self.topology = topology
        
        # all intersections
        self.intersections: Dict[int, Intersection] = {}
        for i in range(num_intersections):
            self.intersections[i] = Intersection(
                intersection_id=i,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                green_ns=initial_green,
                green_ew=initial_green
            )
        
        # set up connections based on topology
        self._setup_connections()
    
    def _setup_connections(self):
        """
        Set up outgoing connections between intersections based on topology and spatial layout.
        
        For a 2×2 grid network (default config):
            [0]----[1]
            |      |
            [2]----[3]
        
        Spatial positions (x, y):
            0: (0, 1)  1: (1, 1)
            2: (0, 0)  3: (1, 0)
        
        Direction mapping:
            - East (+x) from 0 → 1, from 2 → 3
            - West (-x) from 1 → 0, from 3 → 2
            - South (-y) from 0 → 2, from 1 → 3
            - North (+y) from 2 → 0, from 3 → 1
        """

        # check if custom positions were set
        if hasattr(self, 'positions'):
            positions = self.positions
        # Linear corridor (e.g., 4 intersections in a line)
        elif self.num_intersections == 4 and self.topology == {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}:
            positions = {
                0: (0, 0),  # Leftmost
                1: (1, 0),
                2: (2, 0),
                3: (3, 0)   # Rightmost
            }
        # 2×2 grid layout by default
        elif self.num_intersections == 4:
            positions = {
                0: (0, 1),  # Top-left
                1: (1, 1),  # Top-right
                2: (0, 0),  # Bottom-left
                3: (1, 0)   # Bottom-right
            }
        elif self.num_intersections == 9:
            # 3×3 grid
            positions = {
                0: (0, 2), 1: (1, 2), 2: (2, 2),
                3: (0, 1), 4: (1, 1), 5: (2, 1),
                6: (0, 0), 7: (1, 0), 8: (2, 0)
            }
        elif self.num_intersections == 1:
            # single intersection - no connections
            positions = {0: (0, 0)}
        else:
            # auto-detect based on topology or fallback to linear
            positions = self._auto_detect_layout()
        
        # map connections to directions based on spatial positions
        for int_id, connected_ids in self.topology.items():
            if int_id not in self.intersections:
                continue
            
            current_pos = positions[int_id]
            connections = {}
            
            for connected_id in connected_ids:
                if connected_id not in positions:
                    continue
                
                target_pos = positions[connected_id]
                
                # calculate direction based on position difference
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                
                # determine direction
                if dx > 0 and dy == 0:
                    direction = 'E'  # East (right)
                elif dx < 0 and dy == 0:
                    direction = 'W'  # West (left)
                elif dx == 0 and dy > 0:
                    direction = 'N'  # North (up)
                elif dx == 0 and dy < 0:
                    direction = 'S'  # South (down)
                else:
                    # diagonal or invalid connection - skip or handle specially
                    continue
                
                connections[direction] = connected_id
            
            self.intersections[int_id].outgoing_connections = connections
    
    def _auto_detect_layout(self) -> Dict[int, Tuple[int, int]]:
        """
        Auto-detect layout based on topology pattern.
        Fallback: arrange in linear corridor.
        """
        return {i: (i, 0) for i in range(self.num_intersections)}

    def set_custom_positions(self, positions: Dict[int, Tuple[int, int]]):
        """
        Set custom spatial positions for intersections.
        
        Args:
            positions: Dictionary mapping intersection_id to (x, y) coordinates.
            
        Example:
            # Linear corridor (4 intersections in a row)
            network.set_custom_positions({
                0: (0, 0),
                1: (1, 0),
                2: (2, 0),
                3: (3, 0)
            })
        """
        self.positions = positions
        self._setup_connections()  # recalculate connections
        
    def reset_all_stats(self):
        """Reset statistics for all intersections in the network."""
        for intersection in self.intersections.values():
            intersection.reset_stats()
    
    def get_all_lanes(self) -> List[Lane]:
        """
        Get a flat list of all lanes in the network.
        
        Returns:
            List[Lane]: All lanes across all intersections.
        """
        lanes = []
        for intersection in self.intersections.values():
            lanes.extend(intersection.lanes.values())
        return lanes
    
    def update_signal_timings(self, timings: np.ndarray):
        """
        Update traffic light timings from optimizer.
        
        Args:
            timings (np.ndarray): Array of green times in format:
                [int0_NS, int0_EW, int1_NS, int1_EW, int2_NS, int2_EW, ...]
                Length must be 2 * num_intersections.
        
        Raises:
            ValueError: If timings array has incorrect length.
        """
        expected_length = self.num_intersections * 2
        if len(timings) != expected_length:
            raise ValueError(
                f"Timings array must have length {expected_length}, got {len(timings)}"
            )
        
        for i, intersection in self.intersections.items():
            green_ns = timings[i * 2]
            green_ew = timings[i * 2 + 1]
            intersection.traffic_light.update_timings(green_ns, green_ew)
    
    def get_network_metrics(self) -> Dict:
        """
        Calculate aggregate metrics across entire network.
        
        Returns:
            Dict: Network-wide performance metrics.
        """
        all_lanes = self.get_all_lanes()
        
        total_waiting_time = sum(lane.total_waiting_time for lane in all_lanes)
        total_vehicles = sum(lane.total_vehicles_served for lane in all_lanes)
        max_queue = max((lane.current_queue_length for lane in all_lanes), default=0)
        avg_queue = np.mean([lane.current_queue_length for lane in all_lanes])
        
        return {
            'avg_waiting_time': total_waiting_time / total_vehicles if total_vehicles > 0 else 0,
            'max_queue_length': max_queue,
            'avg_queue_length': avg_queue,
            'total_vehicles_served': total_vehicles,
            'num_lanes': len(all_lanes)
        }
    
    def __repr__(self):
        return (f"Network({self.num_intersections} intersections, "
                f"{len(self.get_all_lanes())} total lanes)")

    def get_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get spatial positions of all intersections."""
        if hasattr(self, 'positions'):
            return self.positions
        
        # return default positions based on network size
        if self.num_intersections == 4:
            return {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}
        elif self.num_intersections == 9:
            return {
                0: (0, 2), 1: (1, 2), 2: (2, 2),
                3: (0, 1), 4: (1, 1), 5: (2, 1),
                6: (0, 0), 7: (1, 0), 8: (2, 0)
            }
        else:
            return {i: (i, 0) for i in range(self.num_intersections)}


if __name__ == "__main__":
    """Unit tests for network module"""

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def test_lane_creation():
        """Test Lane class initialization and validation"""
        print("Testing Lane creation...", end=" ")
        
        # valid lane
        lane = Lane(0, 'N', 0.2, 0.4)
        assert lane.direction == 'N'
        assert lane.utilization == 0.5
        assert lane.theoretical_avg_waiting_time == 5.0
        
        # test validation
        try:
            Lane(0, 'X', 0.2, 0.4)  # invalid direction
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        
        try:
            Lane(0, 'N', -0.2, 0.4)  # negative arrival rate
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_traffic_light():
        """Test TrafficLight class"""
        print(f"Testing TrafficLight...{RESET}", end=" ")
        
        light = TrafficLight(green_ns=30, green_ew=40)
        assert light.green_ns == 30
        assert light.green_ew == 40
        assert light.cycle_time == 30 + 40 + 2*3  # 76 seconds
        assert light.is_green('N') == True
        assert light.is_green('E') == False
        
        # test update
        light.update_timings(50, 35)
        assert light.green_ns == 50
        assert light.cycle_time == 50 + 35 + 6
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_intersection():
        """Test Intersection class"""
        print("Testing Intersection...", end=" ")
        
        intersection = Intersection(0, 0.2, 0.4, 30, 30)
        assert intersection.intersection_id == 0
        assert len(intersection.lanes) == 4
        assert 'N' in intersection.lanes
        assert intersection.lanes['N'].direction == 'N'
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_2x2_grid_connections():
        """Test 2×2 grid network connections"""
        print("Testing 2×2 grid connections...", end=" ")
        
        network = Network(
            num_intersections=4,
            topology={
                0: [1, 2],
                1: [0, 3],
                2: [0, 3],
                3: [1, 2]
            },
            arrival_rate=0.2,
            service_rate=0.4,
            initial_green=30
        )
        
        # verify connections
        assert network.intersections[0].outgoing_connections == {'E': 1, 'S': 2}, \
            f"Int 0 connections wrong: {network.intersections[0].outgoing_connections}"
        assert network.intersections[1].outgoing_connections == {'W': 0, 'S': 3}, \
            f"Int 1 connections wrong: {network.intersections[1].outgoing_connections}"
        assert network.intersections[2].outgoing_connections == {'N': 0, 'E': 3}, \
            f"Int 2 connections wrong: {network.intersections[2].outgoing_connections}"
        assert network.intersections[3].outgoing_connections == {'N': 1, 'W': 2}, \
            f"Int 3 connections wrong: {network.intersections[3].outgoing_connections}"
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_linear_corridor_connections():
        """Test linear corridor network connections"""
        print("Testing linear corridor connections...", end=" ")
        
        network = Network(
            num_intersections=4,
            topology={
                0: [1],
                1: [0, 2],
                2: [1, 3],
                3: [2]
            },
            arrival_rate=0.2,
            service_rate=0.4,
            initial_green=30
        )
        
        # verify connections
        assert network.intersections[0].outgoing_connections == {'E': 1}, \
            f"Int 0 connections wrong: {network.intersections[0].outgoing_connections}"
        assert network.intersections[1].outgoing_connections == {'W': 0, 'E': 2}, \
            f"Int 1 connections wrong: {network.intersections[1].outgoing_connections}"
        assert network.intersections[2].outgoing_connections == {'W': 1, 'E': 3}, \
            f"Int 2 connections wrong: {network.intersections[2].outgoing_connections}"
        assert network.intersections[3].outgoing_connections == {'W': 2}, \
            f"Int 3 connections wrong: {network.intersections[3].outgoing_connections}"
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_3x3_grid_connections():
        """Test 3×3 grid network connections"""
        print("Testing 3×3 grid connections...", end=" ")
        
        network = Network(
            num_intersections=9,
            topology={
                0: [1, 3],
                1: [0, 2, 4],
                2: [1, 5],
                3: [0, 4, 6],
                4: [1, 3, 5, 7],
                5: [2, 4, 8],
                6: [3, 7],
                7: [4, 6, 8],
                8: [5, 7]
            },
            arrival_rate=0.2,
            service_rate=0.4,
            initial_green=30
        )
        
        # check center intersection (4) - should connect in all 4 directions
        assert network.intersections[4].outgoing_connections == {'N': 1, 'W': 3, 'E': 5, 'S': 7}, \
            f"Int 4 (center) connections wrong: {network.intersections[4].outgoing_connections}"
        
        # check corner (0) - should only connect E and S
        assert network.intersections[0].outgoing_connections == {'E': 1, 'S': 3}, \
            f"Int 0 (corner) connections wrong: {network.intersections[0].outgoing_connections}"
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_network_methods():
        """Test Network helper methods"""
        print("Testing Network methods...", end=" ")
        
        network = Network(4, config.NETWORK_TOPOLOGY, 0.2, 0.4, 30)
        
        # test get_all_lanes
        lanes = network.get_all_lanes()
        assert len(lanes) == 16, f"Expected 16 lanes, got {len(lanes)}"
        
        # test update_signal_timings
        new_timings = np.array([35, 40, 25, 50, 30, 45, 40, 35])
        network.update_signal_timings(new_timings)
        assert network.intersections[0].traffic_light.green_ns == 35
        assert network.intersections[0].traffic_light.green_ew == 40
        assert network.intersections[1].traffic_light.green_ns == 25
        
        # test invalid timings array
        try:
            network.update_signal_timings(np.array([30, 30]))  # too short
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Timings array must have length 8" in str(e)
        
        # test get_network_metrics (with zero traffic)
        metrics = network.get_network_metrics()
        assert 'avg_waiting_time' in metrics
        assert 'max_queue_length' in metrics
        assert metrics['num_lanes'] == 16
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_custom_positions():
        """Test custom position setting"""
        print("Testing custom positions...", end=" ")
        
        network = Network(3, {0: [1], 1: [0, 2], 2: [1]}, 0.2, 0.4, 30)
        
        # set custom diagonal layout
        network.set_custom_positions({
            0: (0, 0),
            1: (1, 1),
            2: (2, 2)
        })
        
        # check connections were recalculated
        # from (0,0) to (1,1) is diagonal, should be skipped
        # so no connections should exist
        assert len(network.intersections[0].outgoing_connections) == 0, \
            "Diagonal connections should be skipped"
        
        # set proper linear layout
        network.set_custom_positions({
            0: (0, 0),
            1: (1, 0),
            2: (2, 0)
        })
        
        assert network.intersections[0].outgoing_connections == {'E': 1}
        assert network.intersections[1].outgoing_connections == {'W': 0, 'E': 2}
        
        print(f"{GREEN}PASSED{RESET}")
    
    def test_mm1_theory_validation():
        """Test M/M/1 theoretical calculations"""
        print("Testing M/M/1 theory...", end=" ")
        
        # λ=0.2, μ=0.4
        lane = Lane(0, 'N', 0.2, 0.4)
        
        # ρ = λ/μ = 0.5
        assert abs(lane.utilization - 0.5) < 0.001
        
        # L = ρ/(1-ρ) = 0.5/0.5 = 1.0
        assert abs(lane.theoretical_avg_queue_length - 1.0) < 0.001
        
        # W = 1/(μ-λ) = 1/0.2 = 5.0
        assert abs(lane.theoretical_avg_waiting_time - 5.0) < 0.001
        
        # test unstable queue (ρ >= 1)
        unstable = Lane(1, 'S', 0.5, 0.4)  # ρ = 1.25
        assert unstable.theoretical_avg_queue_length == float('inf')
        assert unstable.theoretical_avg_waiting_time == float('inf')
        
        print(f"{GREEN}PASSED{RESET}")
    
    print("="*60)
    print("NETWORK MODULE UNIT TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_lane_creation,
        test_traffic_light,
        test_intersection,
        test_2x2_grid_connections,
        test_linear_corridor_connections,
        test_3x3_grid_connections,
        test_network_methods,
        test_custom_positions,
        test_mm1_theory_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{RED} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"{RED} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print(f"{GREEN}RESULTS: {passed} passed, {failed} failed{RESET}")
    else:
        print(f"RESULTS: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")
    print("="*60)
    
    if failed == 0:
        print(f"\n{GREEN}All tests passed! Network module is working correctly.{RESET}\n")
    else:
        print(f"\n{RED}{failed} test(s) failed. Fix issues above.{RESET}\n")
        exit(1)