#!/usr/bin/env python3

import search


# Define a class for solving the Fleet Problem
class FleetProblem(search.Problem):
    def __init__(self):
        # Initialize instance variables
        self.sol = None  # Store the solution
        self.graph = None  # Store the graph data
        self.requests = None  # Store the request data
        self.vehicles = None  # Store the vehicle data

    # Load data from a file into the FleetProblem instance
    def load(self, file_content):
        current_mode = None  # Track the current mode (P, R, or V)
        row = 0  # Track the current row being processed
        P = 0

        for line in file_content:
            words = line.split()  # Split the line into words

            if not words or words[0].startswith('#'):
                continue  # Skip empty lines and comments

            if words[0] == 'P':
                P = int(words[1])  # Get the number of vertices
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
                elif current_mode == 'V':
                    self.vehicles.add_vehicle(words)  # Add vehicle data
                else:
                    raise Exception('Invalid mode')  # Handle invalid mode

    # Calculate the cost of a solution
    def cost(self, sol):
        total_cost = 0
        for action in sol:
            if action[0] == 'Dropoff':
                request = self.requests.get_request(action[2])
                td = action[3]
                t = request[0]
                Tod = self.graph.get_edge(request[1], request[2])
                total_cost += td - t - Tod  # Calculate cost based on the solution

        return total_cost


# Define a class for representing a graph
class Graph:
    def __init__(self, num_vertices, directed=False):
        self.num_vertices = num_vertices
        self.directed = directed
        self.graph = [[0] * num_vertices for _ in range(num_vertices)]  # Initialize adjacency matrix

    # Add an edge to the graph
    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        if not self.directed:
            self.graph[v][u] = w

    # Get the weight of an edge
    def get_edge(self, u, v):
        return self.graph[u][v]

    # Print the graph
    def print_graph(self):
        print('Graph:')
        for row in self.graph:
            print(row)
        print()


# Define a class for representing requests
class Requests:
    def __init__(self, num_requests):
        self.num_requests = num_requests
        self.requests = []

    # Add a request to the list of requests
    def add_request(self, words):
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

    # Get a request by index
    def get_request(self, index):
        return self.requests[index]

    # Print the list of requests
    def print_requests(self):
        print('Requests:')
        for request in self.requests:
            print(request)
        print()


# Define a class for representing vehicles
class Vehicles:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.vehicles = []

    # Add a vehicle to the list of vehicles
    def add_vehicle(self, words):
        vehicle = int(words[0])
        self.vehicles.append(vehicle)

    # Get a vehicle by index
    def get_vehicle(self, index):
        return self.vehicles[index]

    # Print the list of vehicles
    def print_vehicles(self):
        print('Vehicles:')
        for vehicle in self.vehicles:
            print(vehicle)
        print()


if __name__ == '__main__':
    sols = [('Pickup', 0, 3, 30.0), ('Pickup', 1, 4, 25.0), ('Pickup', 1, 0, 25.0),
            ('Dropoff', 1, 4, 75.0), ('Dropoff', 1, 0, 75.0), ('Pickup', 0, 2, 30.0),
            ('Dropoff', 0, 3, 80.0), ('Pickup', 0, 1, 80.0), ('Dropoff', 0, 1, 140.0),
            ('Dropoff', 0, 2, 140.0)]

    fp = FleetProblem()
    file_path = 'test.txt'

    with open(file_path, 'r') as fh:
        fp.load(fh)

    print('Total delay:', fp.cost(sols))
