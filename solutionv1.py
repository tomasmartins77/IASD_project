#!/usr/bin/env python3

import search
import copy


# Define a class for solving the Fleet Problem
def get_pickups(requests, vehicle):
    """Gets the list of pickups in a given node.

    Args:
        requests (list): The list of requests.
        vehicle (Vehicle): The vehicle to get the pickups from.

    Returns:
        list: The list of pickups in the given node.
    """
    pickups = []

    for request in requests:
        request_index = request.get_index()
        num_passengers = request.get_num_passengers()
        if request.status == 'waiting' and not vehicle.becomes_full(num_passengers):
            # Check if the request is waiting and the vehicle does not become full after adding the passengers
            pickups.append(request_index)

    return pickups


def get_dropoffs(requests, vehicle):
    """Gets the list of dropoffs in a given node.

    Args:
        requests (list): The list of requests.
        vehicle (Vehicle): The vehicle to get the dropoffs from.

    Returns:
        list: The list of dropoffs in the given node.
    """
    dropoffs = []

    current_requests = vehicle.get_current_requests()  # Get the current requests of the vehicle

    for request in requests:
        request_index = request.get_index()
        if request.status == 'traveling' and request_index in current_requests:
            # Check if the request is in the current requests of the vehicle and traveling
            dropoffs.append(request_index)

    return dropoffs


def print_requests(requests):
    """Prints the requests."""
    print('Requests:')
    for request in requests:
        request.print_request()
    print()


def print_vehicles(vehicles):
    """Prints the vehicles."""
    print('Vehicles:')
    for vehicle in vehicles:
        vehicle.print_vehicle()
    print()


def pickup(state, action):
    """Picks up a request.

        Args:
            state (list): The state to pick up the request from.
            action (list): The action to perform.

        Returns:
            list: The new state after picking up the request.
    """
    action_vehicle = state[0][action[1]]  # Get the vehicle from the state
    which_pickup = action[2]  # Get the index of the request to pick up
    request = state[1][which_pickup]  # Get the request to pick up

    num_passengers = request.get_num_passengers()  # Get the number of passengers
    pickup_location = request.get_pickup()  # Get the pickup location

    action_vehicle.add_passengers(num_passengers)  # Add the passengers to the vehicle
    action_vehicle.add_new_request(which_pickup)  # Add the request to the vehicle
    action_vehicle.set_location(pickup_location)  # Set the location of the vehicle

    request.pick_request()  # Pick up the request
    action_vehicle.set_time(action[3])  # Set the time of the vehicle

    return state


def dropoff(state, action):
    """Drops off a request.

        Args:
            state (list): The state to drop off the request from.
            action (list): The action to perform.

        Returns:
            list: The new state after dropping off the request.
    """
    action_vehicle = state[0][action[1]]  # Get the vehicle from the state
    which_dropoff = action[2]  # Get the index of the request to drop off
    dropoff_request = state[1][which_dropoff]  # Get the request to drop off

    num_passengers = dropoff_request.get_num_passengers()  # Get the number of passengers
    dropoff_location = dropoff_request.get_dropoff()  # Get the dropoff location

    action_vehicle.remove_passengers(num_passengers)  # Remove the passengers from the vehicle
    action_vehicle.remove_request(which_dropoff)  # Remove the request from the vehicle

    action_vehicle.set_time(action[3])  # Set the time of the vehicle
    dropoff_request.drop_request()  # Drop off the request
    action_vehicle.set_location(dropoff_location)  # Set the location of the vehicle

    return state


class FleetProblem(search.Problem):
    """A class for solving the Fleet Problem.

    Attributes:
        sol (list): The solution to the problem.
        graph (Graph): The graph data.
        requests (Requests): The request data.
        vehicles (Vehicles): The vehicle data.
    """

    def __init__(self):
        """Initializes a FleetProblem instance."""
        self.sol = []
        self.graph = None
        self.requests = []
        self.vehicles = []
        self.initial = None
        self.number_of_nodes = 0
        self.total_requests = 0
        self.total_vehicles = 0

    def load(self, file_content):
        """Loads data from a file into the FleetProblem instance.

        Args:
            file_content (list): The contents of the file to load.

        Raises:
            Exception: If an invalid mode is encountered.
        """

        current_mode = None  # Track the current mode (P, R, or V)
        row = 0  # Track the current row being processed
        P = 0
        number_of_requests = 0
        number_of_vehicles = 0

        for line in file_content:
            words = line.split()  # Split the line into words

            if not words or words[0].startswith('#'):
                continue  # Skip empty lines and comments

            if words[0] in ['P', 'R', 'V']:
                current_mode = words[0]
                if current_mode == 'P':
                    P = self.number_of_nodes = int(words[1])
                    self.graph = Graph(P)  # Initialize the graph
                elif current_mode == 'R':
                    self.total_requests = int(words[1])  # Initialize requests
                elif current_mode == 'V':
                    self.total_vehicles = int(words[1])  # Initialize vehicles
            else:
                if current_mode == 'P':
                    col = 1 - P  # Initialize column index for graph data
                    for weight in words:
                        self.graph.add_edge(row, col, float(weight))  # Add edge data to the graph
                        col += 1
                    row += 1
                    P -= 1
                elif current_mode == 'R':
                    self.requests.append(Request(number_of_requests, words))  # Add request data
                    number_of_requests += 1
                elif current_mode == 'V':
                    new_vehicle = Vehicle(number_of_vehicles, int(words[0]))
                    self.vehicles.append(new_vehicle)  # Add vehicle data
                    number_of_vehicles += 1  # Add vehicle data
                else:
                    raise Exception('Invalid mode')  # Handle invalid mode

        self.requests = tuple(self.requests)
        self.vehicles = tuple(self.vehicles)
        self.initialize()  # initialize the state

    def cost(self, sol):
        """Calculates the cost of a solution.

        Args:
            sol (list): The solution to calculate the cost for.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0

        for action in sol:
            if action[0] == 'Dropoff':  # If the action is a dropoff
                request = self.requests[action[2]].get_request()
                td = action[3]
                t = request.get_pickup_time()
                Tod = self.graph.get_edge(request.pickup_location, request.dropoff_location)

                total_cost += td - t - Tod  # Calculate cost based on the solution

        return total_cost

    def initialize(self):
        """Initializes the Fleet Problem."""
        vehicles = copy.deepcopy(self.vehicles)
        requests = copy.deepcopy(self.requests)
        self.initial = tuple([tuple(vehicles), tuple(requests)])

    def actions(self, state):
        """Gets the actions that can be performed on a given state.

            Args:
                state (list): The state to get the actions from.

            Returns:
                list: The list of actions that can be performed on the given state.
        """
        actions = []
        # Get the list of requests and vehicles from the state
        vehicles, requests = state

        for vehicle in vehicles:  # For each vehicle
            i = vehicle.get_index()
            vehicle_time = vehicle.get_time()
            vehicle_location = vehicle.get_location()

            # Get the list of possible pickups
            pickups = get_pickups(requests, vehicle)

            # Get the list of possible dropoffs
            dropoffs = get_dropoffs(requests, vehicle)

            # Get the list of possible actions
            for what_dropoff in dropoffs:
                time_to_add = self.graph.get_edge(vehicle_location, requests[what_dropoff].get_dropoff())

                actions.append(('Dropoff', i, what_dropoff, vehicle_time + time_to_add))  # Add the action to the list

            for what_pickup in pickups:
                request = requests[what_pickup]
                request_time = request.get_pickup_time()
                transport_time = vehicle_time + self.graph.get_edge(vehicle_location, request.get_pickup())

                if request_time > transport_time:  # If the request time is greater than the transport time
                    time_to_add = request_time
                else:
                    time_to_add = self.graph.get_edge(vehicle_location, request.get_pickup())

                actions.append(('Pickup', i, what_pickup, vehicle_time + time_to_add))  # Add the action to the list

        return actions

    def result(self, state, action):
        """Gets the result of an action.

            Args:
                state (list): The state to get the result from.
                action (list): The action to perform.

            Returns:
                list: The new state after performing the action.
        """
        new_state = None

        if action[0] == 'Pickup':
            new_state = pickup(copy.deepcopy(state), action)
        elif action[0] == 'Dropoff':
            new_state = dropoff(copy.deepcopy(state), action)

        return tuple(new_state)

    def goal_test(self, state):
        """Checks if a given state is a goal state.

            Args:
                state (list): The state to check.

            Returns:
                bool: True if the state is a goal state, False otherwise.
        """
        return len([request for request in state[1] if request.status == 'completed']) == self.total_requests

    def path_cost(self, c, state1, action, state2):
        """Calculates the path cost of a given action.

            Args:
                c (float): The cost of the path so far.
                state1 (list): The state to get the path cost from.
                action (list): The action to perform.
                state2 (list): The state to get the path cost to.

            Returns:
                float: The path cost of the given action.
        """
        if action[0] == "Dropoff":
            request = state1[1][action[2]].get_request()
            c += action[3] - request.get_pickup_time() - self.graph.get_edge(request.pickup_location,
                                                                             request.dropoff_location)

        return c

    def solve(self):
        """Solves the Fleet Problem.

        Returns:
            list: The solution to the Fleet Problem.
        """
        resulted = search.uniform_cost_search(self)  # Solve the Fleet Problem using uniform cost search

        res = resulted.solution()
        res = [tuple(i) for i in res]

        return res


# Define a class for representing a graph
class Graph:
    """A class for representing a graph.

    Attributes:
        num_vertices (int): The number of vertices in the graph.
        directed (bool): Whether the graph is directed or not.
        graph (list): The adjacency matrix representing the graph.
    """

    def __init__(self, num_vertices, directed=False):
        """Initializes a Graph instance.

        Args:
            num_vertices (int): The number of vertices in the graph.
            directed (bool, optional): Whether the graph is directed or not. Defaults to False.
        """
        self.num_vertices = num_vertices  # The number of vertices in the graph
        self.directed = directed  # Whether the graph is directed or not
        self.graph = [[0.0] * num_vertices for _ in range(num_vertices)]  # The adjacency matrix representing the graph

    def add_edge(self, u, v, w):
        """Adds an edge to the graph.

        Args:
            u (int): The index of the first vertex.
            v (int): The index of the second vertex.
            w (float): The weight of the edge.
        """

        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError("u and v must be integers")
        if not isinstance(w, float):
            raise ValueError("w must be a float")

        self.graph[u][v] = w
        if not self.directed:
            self.graph[v][u] = w

    def get_edge(self, u, v):
        """Gets the weight of an edge.

        Args:
            u (int): The index of the first vertex.
            v (int): The index of the second vertex.

        Returns:
            float: The weight of the edge.
        """
        return self.graph[u][v]

    def print_graph(self):
        """Prints the graph."""
        print('Graph:')
        for row in self.graph:
            print(row)
        print()


class Request:
    """A class for representing a request.

        Attributes:
            request_index (int): The index of the request.
            pickup_time (float): The time at which the request is made.
            pickup_location (int): The location of the pickup.
            dropoff_location (int): The location of the dropoff.
            n_passengers (int): The number of passengers.
            status (str): The status of the request.
    """

    def __init__(self, request_index, request):
        self.request_index = request_index  # The index of the request
        self.pickup_time = float(request[0])  # The time at which the request is made
        self.pickup_location = int(request[1])  # The location of the pickup
        self.dropoff_location = int(request[2])  # The location of the dropoff
        self.n_passengers = int(request[3])  # The number of passengers
        self.status = 'waiting'  # The status of the request

    def get_request(self):
        """Gets the request."""
        return self

    def get_index(self):
        """Gets the index of the request."""
        return self.request_index

    def get_pickup_time(self):
        """Gets the pickup time."""
        return self.pickup_time

    def get_pickup(self):
        """Gets the pickup location."""
        return self.pickup_location

    def get_dropoff(self):
        """Gets the dropoff location."""
        return self.dropoff_location

    def get_num_passengers(self):
        """Gets the number of passengers."""
        return self.n_passengers

    def get_status(self):
        """Gets the status of the request."""
        return self.status

    def pick_request(self):
        """Picks up a request."""
        self.status = 'traveling'

    def drop_request(self):
        """Drops off a request."""
        self.status = 'completed'

    def print_request(self):
        print('R: ', self.request_index, 'PickT: ', self.pickup_time, 'PickL: ',
              self.pickup_location, 'DropL: ', self.dropoff_location, 'NumP: ',
              self.n_passengers, 'Stat: ', self.status)


class Vehicle:
    """A class for representing a vehicle.

        Attributes:
            vehicle_index (int): The index of the vehicle.
            current_location (int): The current location of the vehicle.
            occupancy (int): The maximum occupancy of the vehicle.
            passengers (int): The number of passengers in the vehicle.
            current_requests (list): The list of requests currently in the vehicle.
    """

    def __init__(self, vehicle_index, occupancy):
        self.vehicle_index = vehicle_index  # The index of the vehicle
        self.current_location = 0  # The current location of the vehicle
        self.occupancy = occupancy  # The maximum occupancy of the vehicle
        self.passengers = 0  # The number of passengers in the vehicle
        self.current_requests = []  # The list of requests currently in the vehicle
        self.current_time = 0  # The current time of the vehicle

    def __lt__(self, other):
        # replace 'attribute' with the actual attribute you want to compare
        return self.current_time < other.current_time

    def get_vehicle(self):
        """Gets a vehicle."""
        return self

    def get_index(self):
        """Gets the index of a vehicle."""
        return self.vehicle_index

    def get_time(self):
        """Gets the time of a vehicle."""
        return self.current_time

    def add_time(self, time):
        """Adds time to a vehicle."""
        self.current_time += time

    def set_time(self, time):
        """Sets the time of a vehicle."""
        self.current_time = time

    def get_current_requests(self):
        """Gets the current requests of a vehicle."""
        return self.current_requests

    def add_new_request(self, request):
        """Adds a new request to a vehicle."""
        self.current_requests.append(request)

    def remove_request(self, request):
        """Removes a request from a vehicle."""
        if request in self.current_requests:
            self.current_requests.remove(request)
        else:
            raise Exception('Request not found')

    def get_location(self):
        """Gets the location of a vehicle."""
        return self.current_location

    def set_location(self, location):
        """Sets the location of a vehicle."""
        self.current_location = location

    def get_occupancy(self):
        """Gets the occupancy of a vehicle."""
        return self.occupancy

    def becomes_full(self, n_passengers):
        """Checks if a vehicle becomes full after adding a given number of passengers.

            Args:
                n_passengers (int): The number of passengers to add.

            Returns:
                bool: True if the vehicle becomes full, False otherwise.
        """
        return self.passengers + n_passengers > self.occupancy

    def add_passengers(self, n_passengers):
        """Adds passengers to a vehicle.

        Args:
            :param n_passengers: number of passengers to add
        """
        self.passengers += n_passengers

    def remove_passengers(self, n_passengers):
        """Removes passengers from a vehicle.

        Args:
            :param n_passengers: number of passengers to remove
        """
        self.passengers -= n_passengers

    def print_vehicle(self):
        print('V: ', self.vehicle_index, 'CurrL: ', self.current_location, 'MAX: ',
              self.occupancy, 'Pass: ', self.passengers, 'CurrR: ', self.current_requests, 'Time: ', self.current_time)


if __name__ == '__main__':
    fp = FleetProblem()
    file_path = 'ex0.dat'
    with open(file_path) as f:
        fp.load(f.readlines())

    reso = fp.solve()
    cost = fp.cost(reso)
    print(reso)
    print(cost)
