#!/usr/bin/env python3

from search import Problem, uniform_cost_search, depth_first_tree_search, depth_first_graph_search, breadth_first_tree_search, breadth_first_graph_search
import copy

class FleetProblem(Problem):

    def __init__(self, initial=None, goal=None):
        super().__init__(initial, goal)
        self.requests = []
        self.fleet = []
        self.graph = None

    def set_requests(self, requests):
        self.requests = requests

    def set_fleet(self, fleet):
        self.fleet = fleet

    def set_graph(self, graph):
        self.graph = graph

    def load(self, file):

        current_mode = None
        row = 0
        P = 0

        with open(file) as f:
            for line in f:
                words = line.split()

                if not words or words[0].startswith('#'):
                    continue  # Skip empty lines and comments

                if words[0] == 'P':
                    P = int(words[1])  # Get the number of vertices
                    self.graph = Graph(P)  # Initialize the graph
                    current_mode = 'P'  # Set the mode to process graph data
                elif words[0] == 'R':
                    total_requests = int(words[1])  # Initialize requests
                    current_mode = 'R'  # Set the mode to process request data
                elif words[0] == 'V':
                    total_vehicles = int(words[1])  # Initialize vehicles
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
                        self.requests.append(Request(float(words[0]), int(words[1]), int(words[2]), int(words[3])))  # Add request data
                    elif current_mode == 'V':
                        self.fleet.append(Vehicle(int(words[0])))  # Add vehicle data
                    else:
                        raise Exception('Invalid mode')  # Handle invalid mode

        self.requests = tuple(self.requests)
        self.fleet = tuple(self.fleet)

        self.initialize()
                    
    def initialize(self):
        self.initial = State(self.requests, self.fleet)

    def actions(self, state):

        actions = []

        requests = state.get_requests()
        vehicles = state.get_vehicles()
        for request in self.requests:
            request.print_request()
        for vehicle in vehicles:
            for request in requests:
                if request.get_status() == 'waiting' and vehicle.get_capacity() >= request.get_passengers():
                    time = vehicle.get_time()
                    deslocation_time = self.graph.get_edge(vehicle.get_location(), request.get_pickup())
                    arrival_time = time + deslocation_time
                    request_time = request.get_time()
                    pickup_time = max(arrival_time, request_time)

                    actions.append(Action('Pickup', vehicles.index(vehicle), requests.index(request), pickup_time))
                if request.get_status() == 'traveling' and request.get_vehicle_id() == vehicles.index(vehicle):
                    time = vehicle.get_time()
                    #print("request adsasdasd")
                    request.print_request()
                    #print("accao", request.get_dropoff(), requests.index(request), request.get_pickup_time())
                    deslocation_time = self.graph.get_edge(vehicle.get_location(), request.get_dropoff())
                    dropoff_time = time + deslocation_time
                    #print("accao", 'Dropoff', vehicles.index(vehicle), requests.index(request), dropoff_time)
                    actions.append(Action('Dropoff', vehicles.index(vehicle), requests.index(request), dropoff_time))

        # actions.sort(key=lambda x: x.get_time())

        # if len(actions) >= 3:
        #     actions = actions[:3]
        # test = dict()
        for actionss in actions:
            print(actionss.print_action())
        print('-------------------')
        return actions
    
    def result(self, state, action):
        new_state = copy.deepcopy(state)

        type = action.get_type()
        vehicle = new_state.get_vehicles()[action.get_vehicle_id()]
        request = new_state.get_requests()[action.get_request_id()]

        if type == 'Pickup':
            request.pick_request(action.get_vehicle_id(), action.get_time())
            vehicle.pick_passengers(request.get_passengers(), action.get_request_id())
            vehicle.set_location(request.get_pickup())
            vehicle.set_time(action.get_time())

        if type == 'Dropoff':
            request.drop_request(action.get_time())
            vehicle.drop_passengers(request.get_passengers(), action.get_request_id())
            vehicle.set_location(request.get_dropoff())
            vehicle.set_time(action.get_time())

        return new_state
    
    def goal_test(self, state):
        requests = state.get_requests()
        for request in requests:
            if request.get_status() != 'completed':
                return False
        return True
    
    def path_cost(self, c, state1, action, state2):

        vehicle = state1.get_vehicles()[action.get_vehicle_id()]
        request = state1.get_requests()[action.get_request_id()]
        time = action.get_time()

        # if action.get_type() == 'Pickup' and action.get_vehicle_id() == 1 and action.get_request_id() == 1 and action.get_time() == 80.0:
        #     print('Yessssss')
        # if action.get_type() == 'Dropoff' and action.get_vehicle_id() == 1 and action.get_request_id() == 1 and action.get_time() == 140.0:
        #     print('Siuuuuuu')
        
        alpha = 1
        beta = 1

        locations = []

        # for request_id in vehicle.get_requests():
            # if request_id != action.get_request_id():
            #     queue_request = state1.get_requests()[request_id]
            #     pickup_time = queue_request.get_time()

            #     # if queue_request.get_dropoff() not in locations:
            #     #     locations.append(queue_request.get_dropoff())

            #     if action.get_type() == 'Pickup':
            #         if queue_request.get_dropoff() != request.get_pickup():
            #             c += time - vehicle.get_time()

            #     if action.get_type() == 'Dropoff':
            #         if queue_request.get_dropoff() != request.get_dropoff():
            #             c += time - vehicle.get_time()

                # c += (time - pickup_time) * queue_request.get_passengers()

        if action.get_type() == 'Pickup':
            # if len(locations) != 0:
            #     if request.get_pickup() not in locations or request.get_dropoff() not in locations:
            #         c += 20
            # c += self.graph.get_edge(vehicle.get_location(), request.get_pickup())

            # if vehicle.get_location() != request.get_pickup():
            #     alpha = 0.8

            return c + (time - request.get_time()) * alpha

            # other_vehicles = state1.get_vehicles()
            # for other in other_vehicles:
            #     if other != vehicle:
            #         if other.get_location() == request.get_pickup():
            #             c += 50
        else:
            # if request.get_dropoff() not in locations and len(locations) != 0:
            #     beta = 1.2
            return c + time - self.graph.get_edge(vehicle.get_location(), request.get_dropoff()) - request.get_pickup_time()


        
        # if action.get_type() == 'Pickup':
        #     if request.get_pickup() not in locations or request.get_dropoff() not in locations:   
        #         c += self.graph.get_edge(request.get_pickup(), request.get_dropoff())
        #     # if request.get_pickup() != vehicle.get_location():
        #     #     c += self.graph.get_edge(vehicle.get_location(), request.get_pickup())
        #     c += self.graph.get_edge(vehicle.get_location(), request.get_pickup())
        # else:
        #     if request.get_dropoff() not in locations:
        #         c += self.graph.get_edge(request.get_pickup(), request.get_dropoff())
        #     c += self.graph.get_edge(vehicle.get_location(), request.get_dropoff()) 
        # return c
    
    def solve(self):
        result = uniform_cost_search(self, True)
        return result.solution()

    def cost(self, solution):
        """Calculates the cost of a solution.

        Args:
            sol (list): The solution to calculate the cost for.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0
        for action in solution:
            if action.get_type() == 'Dropoff':
                request = self.requests[action.get_request_id()]
                td = action.get_time()
                t = request.get_time()
                Tod = self.graph.get_edge(request.get_pickup(), request.get_dropoff())
                total_cost += td - t - Tod  # Calculate cost based on the solution

        return total_cost


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

    def add_edge(self, row, column, weight):
        """Adds an edge to the graph.

        Args:
            row (int): The index of the first vertex.
            column (int): The index of the second vertex.
            weight (float): The weight of the edge.
        """

        if not isinstance(row, int) or not isinstance(column, int):
            raise ValueError("row and column must be integers")
        if not isinstance(weight, float):
            raise ValueError("weight must be a float")

        self.graph[row][column] = weight
        if not self.directed:
            self.graph[column][row] = weight

    def get_edge(self, row, column):
        """Returns the weight of an edge.

        Args:
            row (int): The index of the first vertex.
            column (int): The index of the second vertex.

        Returns:
            float: The weight of the edge.
        """

        if not isinstance(row, int) or not isinstance(column, int):
            raise ValueError("row and column must be integers")

        return self.graph[row][column]
    
    def get_neighbors(self, vertex):
        """Returns the neighbors of a vertex.

        Args:
            vertex (int): The index of the vertex.

        Returns:
            list: The indices of the neighbors of the vertex.
        """

        if not isinstance(vertex, int):
            raise ValueError("vertex must be an integer")

        return [i for i in range(self.num_vertices) if self.graph[vertex][i] != 0]
    
    def get_vertices(self):
        """Returns the indices of the vertices in the graph.

        Returns:
            list: The indices of the vertices in the graph.
        """

        return [i for i in range(self.num_vertices)]\
    
    def get_num_vertices(self):
        """Returns the number of vertices in the graph.

        Returns:
            int: The number of vertices in the graph.
        """

        return self.num_vertices
    
    def get_num_edges(self):
        """Returns the number of edges in the graph.

        Returns:
            int: The number of edges in the graph.
        """

        return sum([1 for i in range(self.num_vertices) for j in range(self.num_vertices) if self.graph[i][j] != 0])
    
    def print_graph(self):
        """Prints the graph.
        """

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                print(self.graph[i][j], end=" ")
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

    def __init__(self, time, pickup, dropoff, passengers):
        self.time = time
        self.pickup = pickup
        self.dropoff = dropoff
        self.passengers = passengers
        self.status = 'waiting'
        self.pickup_time = time
        self.dropoff_time = None
        self.vehicle_id = None
    
    def __eq__(self, other):
        return (    
                self.status == other.status
                and self.vehicle_id == other.vehicle_id
                and self.pickup_time == other.pickup_time
                and self.dropoff_time == other.dropoff_time
            )
    
    def __lt__(self, other):
        return self.time < other.time

    def __hash__(self):
        return hash((self.status, self.vehicle_id, self.pickup_time, self.dropoff_time))

    def get_time(self):
        """Gets the pickup time."""
        return self.time
    
    def get_pickup_time(self):
        """Gets the status time."""
        return self.pickup_time
    
    def get_dropoff_time(self):
        """Gets the status time."""
        return self.drop_time

    def get_pickup(self):
        """Gets the pickup location."""
        return self.pickup

    def get_dropoff(self):
        """Gets the dropoff location."""
        return self.dropoff

    def get_passengers(self):
        """Gets the number of passengers."""
        return self.passengers

    def get_status(self):
        return self.status
    
    def get_vehicle_id(self):
        return self.vehicle_id

    def pick_request(self, vehicle_id, time):
        """Picks up a request."""
        self.vehicle_id = vehicle_id
        self.status = 'traveling'
        self.pickup_time = time

    def drop_request(self, time):
        """Drops off a request."""
        self.status = 'completed'
        self.dropoff_time = time

    def print_request(self):
        print(self.pickup_time, self.pickup, self.dropoff, self.passengers, self.status, self.time)



class Vehicle:
    """A class for representing a vehicle.

        Attributes:
            vehicle_index (int): The index of the vehicle.
            location (int): The location of the vehicle.
            capacity (int): The capacity of the vehicle.
            passengers (list): The passengers in the vehicle.
            status (str): The status of the vehicle.
    """

    def __init__(self, capacity, time = 0, location = 0, passengers = 0):
        self.capacity = capacity
        self.time = time
        self.location = location
        self.passengers = passengers
        self.requests_id = []

    def __lt__(self, other):
        return self.capacity < other.capacity

    def get_capacity(self):
        """Gets the capacity of the vehicle."""
        return self.capacity - self.passengers

    def get_passengers(self):
        """Gets the passengers in the vehicle."""
        return self.passengers

    def get_status(self):
        """Gets the status of the vehicle."""
        return self.status

    def pick_passengers(self, passengers, request_id):
        """Picks up a passenger."""
        if self.passengers + passengers > self.capacity:
            raise Exception('Exceeded capacity')

        self.passengers += passengers
        self.requests_id.append(request_id)

    def drop_passengers(self, passengers, request_id):
        """Drops off a passenger."""
        self.passengers -= passengers
        self.requests_id.remove(request_id)

    def get_requests(self):
        """Gets the requests in the vehicle."""
        return self.requests_id

    def get_time(self):
        """Gets the time of the vehicle."""
        return self.time
    
    def add_time(self, time):
        """Adds time to the vehicle."""
        self.time += time

    def set_time(self, time):
        """Sets the time of the vehicle."""
        self.time = time
    
    def get_location(self):
        """Gets the location of the vehicle."""
        return self.location
    
    def set_location(self, location):
        """Sets the location of the vehicle."""
        self.location = location

class Action:
    def __init__(self, type, vehicle_id, request_id, time):
        self.type = type
        self.vehicle_id = vehicle_id
        self.request_id = request_id
        self.time = time

    def get_type(self):
        return self.type

    def get_vehicle_id(self):
        return self.vehicle_id
    
    def get_request_id(self):
        return self.request_id
    
    def get_time(self):
        return self.time

    def print_action(self):
        print(self.type, self.vehicle_id, self.request_id, self.time)


class State:
    """A class for representing a state.

        Attributes:
            requests (list): The requests in the state.
            vehicles (list): The vehicles in the state.
    """

    def __init__(self, requests, vehicles):
        self.requests = requests
        self.vehicles = vehicles

    def __lt__(self, other):
        return True

    def __hash__(self):
        return self.requests.__hash__()
    
    def __eq__(self, other):
        return self.requests == other.requests

    def get_requests(self):
        """Gets the requests in the state."""
        return self.requests

    def get_vehicles(self):
        """Gets the vehicles in the state."""
        return self.vehicles


if __name__ == '__main__':
    fp = FleetProblem()
    file_path = 'ex0.dat'
    with open(file_path) as f:
        fp.load(f.readlines())

    reso = fp.solve()
    cost = fp.cost(reso)
    print(reso)
    print(cost)
