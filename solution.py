#!/usr/bin/env python3

import search


# Define a class for solving the Fleet Problem
class FleetProblem(search.Problem):
    """A class for solving the Fleet Problem.

    Attributes:
        sol (list): The solution to the problem.
        graph (Graph): The graph data.
        requests (Requests): The request data.
        vehicles (Vehicles): The vehicle data.
    """

    def __init__(self, initial):
        """Initializes a FleetProblem instance."""
        super().__init__(initial)
        self.sol = None
        self.graph = None
        self.requests = None
        self.vehicles = None
        self.initial = []
        self.number_of_nodes = 0

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
                self.requests = Requests(int(words[1]))  # Initialize requests
                current_mode = 'R'  # Set the mode to process request data
            elif words[0] == 'V':
                self.vehicles = Vehicles(int(words[1]))  # Initialize vehicles
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
                    self.requests.add_request(words)  # Add request data
                    self.initialize(words)
                elif current_mode == 'V':
                    self.vehicles.add_vehicle(words[0])  # Add vehicle data
                else:
                    raise Exception('Invalid mode')  # Handle invalid mode

        print("Initial state:")
        print(self.initial)
        print()

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
                request = self.requests.get_request(action[2])
                td = action[3]
                t = request[0]
                Tod = self.graph.get_edge(request[1], request[2])
                total_cost += td - t - Tod  # Calculate cost based on the solution

        return total_cost

    def initialize(self, words):
        """Initializes the problem.

        Args:
            words (list): The words representing the request.
        """
        self.initial.append([0, 0, int(words[1]), int(words[3])])

    def actions(self, state):
        """Gets the actions that can be performed from a given state.

        Args:
            state (list): The state to get the actions for.

        Returns:
            list: The list of actions that can be performed from the given state.
        """
        actions_list = []
        state = [0, 0, 1, 1]

        for i in range(self.number_of_nodes):
            edge = self.graph.get_edge(state[2] - 1, i)
            if edge != 0.0:
                actions_list.append([state[0] + edge, i, state[2], state[3]])

        return actions_list

    def result(self, state, action):
        """Gets the result of performing an action on a given state.

        Args:
            state (list): The state to perform the action on.
            action (list): The action to perform.

        Returns:
            list: The resulting state.
        """
        return [action[0], action[1], state[2], state[3]]

    def goal_test(self, state):
        """Checks if a given state is a goal state.

        Args:
            state (list): The state to check.

        Returns:
            bool: True if the state is a goal state, False otherwise.
        """
        return state[1] == state[2]

    def solve(self):
        """Solves the Fleet Problem.

        Returns:
            list: The solution to the problem.
        """
        search.breadth_first_graph_search(self)


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


class Requests:
    """A class for representing requests.

    Attributes:
        num_requests (int): The number of requests.
        requests (list): The list of requests.
    """

    def __init__(self, num_requests):
        """Initializes a Requests instance.

        Args:
            num_requests (int): The number of requests.
        """
        self.num_requests = num_requests
        self.requests = []

    def add_request(self, words):
        """Adds a request to the list of requests.

        Args:
            words (list): The words representing the request.
        """
        value = 0
        request = []
        for word in words:
            try:
                value = float(word)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
            request.append(value)
        self.requests.append(request)

    def get_request(self, index):
        """Gets a request by index.

        Args:
            index (int): The index of the request.

        Returns:
            list: The request.
        """
        return self.requests[index]

    def print_requests(self):
        """Prints the list of requests."""
        print('Requests:')
        for request in self.requests:
            print(request)
        print()


class Vehicles:
    """A class for representing vehicles.

    Attributes:
        num_vehicles (int): The number of vehicles.
        vehicles (list): The list of vehicles.
    """

    def __init__(self, num_vehicles):
        """Initializes a Vehicles instance.

        Args:
            num_vehicles (int): The number of vehicles.
        """
        self.num_vehicles = num_vehicles
        self.vehicles = []

    def add_vehicle(self, seats):
        """Adds a vehicle to the list of vehicles.

        Args:
            seats (list): The seats representing the vehicle.
        """
        self.vehicles.append(int(seats))

    def get_vehicle(self, index):
        """Gets a vehicle by index.

        Args:
            index (int): The index of the vehicle.

        Returns:
            int: The vehicle.
        """
        return self.vehicles[index]

    def print_vehicles(self):
        """Prints the list of vehicles."""
        print('Vehicles:')
        for vehicle in self.vehicles:
            print(vehicle)
        print()


if __name__ == '__main__':
    fp = FleetProblem([])
    file_path = 'test.txt'
    with open(file_path) as f:
        fp.load(f.readlines())

    fp.graph.print_graph()
    fp.requests.print_requests()
    fp.vehicles.print_vehicles()
    actions = fp.actions(fp.initial)
    states = []
    fp.result(fp.initial[0], actions[0])
