#!/usr/bin/env python3

import search
import copy


# Define a class for solving the Fleet Problem
def get_pickups(requests):
    """Gets the list of pickups in a given node.

    Args:
        requests (list): The list of requests.

    Returns:
        list: The list of pickups in the given node.
    """
    pickups = []
    for request in requests:
        if request.status == 'waiting':
            pickups.append(request.get_index())
    return pickups


def get_dropoffs(requests):
    """Gets the list of dropoffs in a given node.

    Args:
        requests (list): The list of requests.

    Returns:
        list: The list of dropoffs in the given node.
    """
    dropoffs = []
    for request in requests:
        if request.status == 'traveling':
            dropoffs.append(request.get_index())
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
    """
    action_vehicle = state[0][action[1]]
    which_pickup = action[2]
    request = state[1][which_pickup].get_request()

    action_vehicle.add_passengers(request.get_num_passengers())
    action_vehicle.add_new_request(which_pickup)
    action_vehicle.set_location(request.get_pickup())

    request.pick_request()
    action_vehicle.set_time(action[3])

    return state


def where_to_go(state, vehicle):
    go = []
    current_requests = vehicle.get_current_requests()
    dropoff_location = []

    for i in current_requests:
        dropoff_location.append(state[1][i].get_dropoff())

    for request in state[1]:
        if request.get_status() != "completed":
            location = request.get_pickup()

            if location not in go and request.get_status() == "waiting":
                go.append(location)

            location = request.get_dropoff()
            if location not in go and request.get_status() == "traveling" and location in dropoff_location:
                go.append(location)
    return go


def dropoff(state, action):
    """Drops off a request.

        Args:
            state (list): The state to drop off the request from.
            action (list): The action to perform.
    """
    action_vehicle = state[0][action[1]]
    which_dropoff = action[2]
    dropoff_request = state[1][which_dropoff].get_request()

    action_vehicle.remove_passengers(dropoff_request.get_num_passengers())
    action_vehicle.remove_request(which_dropoff)

    action_vehicle.set_time(action[3])
    dropoff_request.drop_request()
    action_vehicle.set_location(dropoff_request.get_dropoff())

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

            if words[0] == 'P':
                P = int(words[1])  # Get the number of vertices
                self.number_of_nodes = P
                self.graph = Graph(P)  # Initialize the graph
                current_mode = 'P'  # Set the mode to process graph data
            elif words[0] == 'R':
                self.total_requests = int(words[1])  # Initialize requests
                current_mode = 'R'  # Set the mode to process request data
            elif words[0] == 'V':
                self.total_vehicles = int(words[1])  # Initialize vehicles
                current_mode = 'V'  # Set the mode to process vehicle data
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
        self.initialize()

    def cost(self, sol):
        """Calculates the cost of a solution.

        Args:
            sol (list): The solution to calculate the cost for.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0
        for action in sol:
            if action[0] == 'Dropoff':
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
        """
        actions = []
        # Get the list of requests and vehicles from the state
        vehicles = state[0]
        requests = state[1]

        for vehicle in vehicles:
            i = vehicle.get_index()

            # Get the list of possible pickups
            pickups = get_pickups(requests)
            # Get the list of possible dropoffs
            dropoffs = get_dropoffs(requests)

            # Get the list of possible actions
            for what_pickup in pickups:
                if not vehicle.becomes_full(requests[what_pickup].get_num_passengers()):
                    request_time = requests[what_pickup].get_pickup_time()
                    transport_time = (vehicle.get_time()
                                      + self.graph.get_edge(vehicle.get_location(), requests[what_pickup].get_pickup()))
                    if request_time > transport_time:
                        time_to_add = requests[what_pickup].get_pickup_time()
                    else:
                        time_to_add = self.graph.get_edge(vehicle.get_location(), requests[what_pickup].get_pickup())

                    actions.append(('Pickup', i, what_pickup, vehicle.get_time() + time_to_add))

            for what_dropoff in dropoffs:
                if what_dropoff in vehicle.get_current_requests():
                    time_to_add = self.graph.get_edge(vehicle.get_location(), requests[what_dropoff].get_dropoff())
                    actions.append(('Dropoff', i, what_dropoff, vehicle.get_time() + time_to_add))

        return actions

    def result(self, state, action):
        """Gets the result of an action.

            Args:
                state (list): The state to get the result from.
                action (list): The action to perform.
        """
        new_state = copy.deepcopy(state)
        if action[0] == 'Pickup':
            new_state = pickup(new_state, action)
        elif action[0] == 'Dropoff':
            new_state = dropoff(new_state, action)

        return tuple(new_state)

    def goal_test(self, state):
        """Checks if a given state is a goal state."""
        done_requests = 0
        for request in state[1]:
            if request.status == 'completed':
                done_requests += 1

        return done_requests == self.total_requests

    def path_cost(self, c, state1, action, state2):
        """Calculates the path cost of a given action."""
        if action[0] == "Dropoff":
            request = state1[1][action[2]].get_request()
            td = action[3]
            t = request.get_pickup_time()
            Tod = self.graph.get_edge(request.pickup_location, request.dropoff_location)
            c += td - t - Tod

        return c

    def solve(self):
        """Solves the Fleet Problem.

        Returns:
            list: The solution to the Fleet Problem.
        """
        resulted = search.uniform_cost_search(self)
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
        self.num_vertices = num_vertices
        self.directed = directed
        self.graph = [[0.0] * num_vertices for _ in range(num_vertices)]

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
        self.request_index = request_index
        self.pickup_time = float(request[0])
        self.pickup_location = int(request[1])
        self.dropoff_location = int(request[2])
        self.n_passengers = int(request[3])
        self.status = 'waiting'

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
        self.vehicle_index = vehicle_index
        self.current_location = 0
        self.occupancy = occupancy
        self.passengers = 0
        self.current_requests = []
        self.current_time = 0

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
    file_path = 'teste1.txt'
    with open(file_path) as f:
        fp.load(f.readlines())

    reso = fp.solve()
    cost = fp.cost(reso)
    print(reso)
    print(cost)
