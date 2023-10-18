#!/usr/bin/env python3

from search import Problem, uniform_cost_search, astar_search
import copy


class FleetProblem(Problem):
    """A class for representing a fleet problem.

        Attributes:
            initial (State): The initial state of the problem.
            goal (State): The goal state of the problem.
            requests (list): The requests of the problem.
            fleet (list): The fleet of the problem.
            graph (Graph): The graph of the problem.
    """

    def __init__(self, initial=None, goal=None):
        super().__init__(initial, goal)
        self.requests = []
        self.fleet = []
        self.graph = None

    def load(self, file):
        """
        Loads the problem from a file.

        Args:
            file (file): The file to load the problem from.
        """
        # Initialize variables
        current_mode = None
        row = 0
        p = 0
        request_index = 0
        vehicle_index = 0

        # Iterate over each line in the file
        for line in file:
            words = line.split()

            # Skip empty lines and comments
            if not words or words[0].startswith('#'):
                continue

            # Process vertices
            if words[0] == 'P':
                p = int(words[1])  # Get the number of vertices
                self.graph = Graph(p)  # Initialize the graph
                current_mode = 'P'  # Set the mode to process graph data

            # Process request data
            elif words[0] == 'R':
                current_mode = 'R'

            # Process vehicle data
            elif words[0] == 'V':
                current_mode = 'V'

            else:
                # Add edge data to the graph
                if current_mode == 'P':
                    col = 1 - p
                    for weight in words:
                        self.graph.add_edge(row, col, float(weight))
                        col += 1
                    row += 1
                    p -= 1

                # Add request data
                elif current_mode == 'R':
                    self.requests.append(Request(float(words[0]), int(words[1]), int(words[2]), int(words[3]),
                                                 request_index))
                    request_index += 1

                # Add vehicle data
                elif current_mode == 'V':
                    self.fleet.append(Vehicle(int(words[0]), vehicle_index))
                    vehicle_index += 1

                else:
                    raise Exception('Invalid mode')  # Handle invalid mode

        self.requests = tuple(self.requests)
        self.fleet = tuple(self.fleet)

        self.initial = State(self.requests, self.fleet)

    def actions(self, state):
        """Returns the actions that can be executed in the given state.

        Args:
            state (State): The state to get the actions for.

        Returns:
            actions (list): The actions that can be executed in the given state.
        """

        actions = []  # Initialize an empty list to store the actions

        # Get the requests and vehicles from the state
        requests = state.get_requests()
        vehicles = state.get_vehicles()

        # Iterate over each vehicle
        for vehicle in vehicles:
            # Iterate over each request
            for request in requests:

                # Check if the request is waiting and if the vehicle has enough capacity
                if request.get_status() == 'waiting' and vehicle.get_capacity() >= request.get_passengers():
                    time = vehicle.get_time()

                    # Calculate the time it takes for the vehicle to reach the pickup location
                    deslocation_time = self.graph.get_edge(vehicle.get_location(), request.get_pickup())
                    arrival_time = time + deslocation_time
                    request_time = request.get_time()
                    pickup_time = max(arrival_time, request_time)

                    # Add a 'Pickup' action to the list of actions
                    actions.append(Action('Pickup', vehicle.get_index(), request.get_index(), pickup_time))

                # Check if the request is traveling and if the vehicle is assigned to this request
                if request.get_status() == 'traveling' and request.get_vehicle_id() == vehicle.get_index():
                    time = vehicle.get_time()

                    # Calculate the time it takes for the vehicle to reach the dropoff location
                    deslocation_time = self.graph.get_edge(vehicle.get_location(), request.get_dropoff())
                    dropoff_time = time + deslocation_time

                    # Add a 'Dropoff' action to the list of actions
                    actions.append(Action('Dropoff', vehicle.get_index(), request.get_index(), dropoff_time))

        # Sort the actions based on their time
        actions.sort(key=lambda x: x.get_time())

        # If there are more than 5 actions, keep only the first 5
        if len(actions) >= 5:
            actions = actions[:5]

        return actions  # Return the list of actions

    def result(self, state, action):
        """
        Returns the state that results from executing the given action in the given state.

        Args:
            state (State): The state to execute the action in.
            action (Action): The action to execute.

        Returns:
            State: The state that results from executing the given action in the given state.
        """
        # Create a deep copy of the state to avoid modifying the original state
        new_state = copy.deepcopy(state)

        # Get the type of action, vehicle and request from the action and state
        action_type = action.get_type()
        vehicle = new_state.get_vehicles()[action.get_vehicle_id()]
        request = new_state.get_requests()[action.get_request_id()]

        # If the action is a 'Pickup' action
        if action_type == 'Pickup':
            # Update the request and vehicle status for pickup
            request.pick_request(action.get_vehicle_id(), action.get_time())
            vehicle.pick_passengers(request.get_passengers(), action.get_request_id())
            vehicle.set_location(request.get_pickup())
            vehicle.set_time(action.get_time())

        # If the action is a 'Dropoff' action
        if action_type == 'Dropoff':
            # Update the request and vehicle status for dropoff
            request.drop_request(action.get_time())
            vehicle.drop_passengers(request.get_passengers(), action.get_request_id())
            vehicle.set_location(request.get_dropoff())
            vehicle.set_time(action.get_time())

        return new_state  # Return the new state after executing the action

    def goal_test(self, state):
        """ Returns True if the given state is a goal state.

            Args:
                state (State): The state to test.

            Returns:
                bool: True if the given state is a goal state, False otherwise.
        """
        requests = state.get_requests()

        for request in requests:
            if request.get_status() != 'completed':
                return False
        return True

    def path_cost(self, c, state1, action, state2):
        """
        Calculates the path cost of an action.

        Args:
            c (float): The cost of the path up to the given action.
            state1 (State): The state before the action.
            action (Action): The action to calculate the path cost for.
            state2 (State): The state after the action.

        Returns:
            float: The path cost of the action.
        """
        # Get the vehicle and request from the first state using the IDs from the action
        vehicle = state1.get_vehicles()[action.get_vehicle_id()]
        request = state1.get_requests()[action.get_request_id()]
        # Get the time from the action
        time = action.get_time()

        # If the action is a 'Pickup' action
        if action.get_type() == 'Pickup':
            # Add to the cost the difference between the time of the action and the time of the request
            c += (time - request.get_time())

            # Get all other vehicles from the first state
            other_vehicles = state1.get_vehicles()
            for other in other_vehicles:
                # If another vehicle is at the pickup location, add 50 to the cost
                if other != vehicle:
                    if other.get_location() == request.get_pickup():
                        c += 50
        else:
            # If it's not a 'Pickup' action, it's a 'Dropoff' action. Subtract from the cost
            # the difference between the time of the action and
            # the time it takes for the vehicle to go from its location to the dropoff location,
            # and subtract also the status time of the request
            c += time - self.graph.get_edge(vehicle.get_location(), request.get_dropoff()) - request.get_status_time()

        return c  # Return the cost

    def h(self, node):
        """Heuristic function for the problem.

        Args:
            node (Node): The node to calculate the heuristic for.

        Returns:
            float: The heuristic value of the node.
        """
        actions = []
        for action in node.solution():
            new = tuple([action.get_type(), action.get_vehicle_id(), action.get_request_id(), action.get_time()])
            actions.append(new)

        return self.cost(actions)  # Return the cost of the solution

    def solve(self):
        """Solves the problem using uniform cost search."""
        result = astar_search(self)
        actions = []

        for action in result.solution():
            new = tuple([action.get_type(), action.get_vehicle_id(), action.get_request_id(), action.get_time()])
            actions.append(new)

        return actions

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

                request = self.requests[action[2]]
                td = action[3]
                t = request.get_time()
                tod = self.graph.get_edge(request.get_pickup(), request.get_dropoff())

                total_cost += td - t - tod  # Calculate cost based on the solution

        return total_cost


class Graph:
    """The class representing the graph.

        Attributes:
            num_vertices (int): The number of vertices in the graph.
            directed (bool, optional): Whether the graph is directed or not. Defaults to False.
            graph (list): The graph.
    """

    def __init__(self, num_vertices, directed=False):

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

        return [i for i in range(self.num_vertices)]

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
        """Prints the graph."""

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                print(self.graph[i][j], end=" ")
            print()


class Request:
    """Initializes a Request instance.

        Attributes:
            time (int): The pickup time of the request.
            pickup (int): The pickup location of the request.
            dropoff (int): The dropoff location of the request.
            passengers (int): The number of passengers of the request.
            status (str): The status of the request.
            status_time (int): The time of the status of the request.
            vehicle_id (int): The id of the vehicle that picked up the request.
            index (int): The index of the request.
    """

    def __init__(self, time, pickup, dropoff, passengers, index):
        self.time = time
        self.pickup = pickup
        self.dropoff = dropoff
        self.passengers = passengers
        self.status = 'waiting'
        self.status_time = time
        self.vehicle_id = None
        self.index = index

    def __eq__(self, other):
        return (
                self.status == other.status
                and self.vehicle_id == other.vehicle_id
                and self.status_time == other.status_time
                and self.index == other.index
        )

    def get_index(self):
        """Gets the index of the request."""
        return self.index

    def __lt__(self, other):
        """Checks if the request is less than another request."""
        return self.time < other.time

    def __hash__(self):
        """ The hash value of the request."""
        return hash((self.status, self.vehicle_id, self.status_time, self.index))

    def get_time(self):
        """Gets the pickup time."""
        return self.time

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
        """Gets the status of the request."""
        return self.status

    def get_status_time(self):
        """Gets the status time of the request."""
        return self.status_time

    def get_vehicle_id(self):
        """Gets the id of the vehicle that picked up the request."""
        return self.vehicle_id

    def pick_request(self, vehicle_id, time):
        """Picks up a request."""
        self.vehicle_id = vehicle_id
        self.status = 'traveling'
        self.status_time = time

    def drop_request(self, time):
        """Drops off a request."""
        self.status = 'completed'
        self.status_time = time


class Vehicle:
    """A class for representing a vehicle.

        Attributes:
            capacity (int): The capacity of the vehicle.
            time (int): The time of the vehicle.
            location (int): The location of the vehicle.
            passengers (int): The passengers in the vehicle.
            requests_id (list): The requests in the vehicle.
            index (int): The index of the vehicle.

    """

    def __init__(self, capacity, index, time=0, location=0, passengers=0):
        self.capacity = capacity
        self.time = time
        self.location = location
        self.passengers = passengers
        self.requests_id = []
        self.index = index

    def get_index(self):
        """Gets the index of the vehicle."""
        return self.index

    def get_capacity(self):
        """Gets the capacity of the vehicle."""
        return self.capacity - self.passengers

    def get_passengers(self):
        """Gets the passengers in the vehicle."""
        return self.passengers

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
    """A class for representing an action.

        Attributes:
            action_type (str): The type of the action.
            vehicle_id (int): The id of the vehicle that executes the action.
            request_id (int): The id of the request that is executed.
            time (int): The time of the action.

    """
    def __init__(self, action_type, vehicle_id, request_id, time):
        self.action_type = action_type
        self.vehicle_id = vehicle_id
        self.request_id = request_id
        self.time = time

    def get_type(self):
        """Gets the type of the action."""
        return self.action_type

    def get_vehicle_id(self):
        """Gets the vehicle id of the action."""
        return self.vehicle_id

    def get_request_id(self):
        """Gets the request id of the action."""
        return self.request_id

    def get_time(self):
        """Gets the time of the action."""
        return self.time


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
        """Checks if the state is less than another state."""
        return True

    def __hash__(self):
        """ The hash value of the requests."""
        return self.requests.__hash__()

    def __eq__(self, other):
        """Checks if two states are equal."""
        return self.requests == other.requests

    def get_requests(self):
        """Gets the requests in the state."""
        return self.requests

    def get_vehicles(self):
        """Gets the vehicles in the state."""
        return self.vehicles


if __name__ == '__main__':
    fp = FleetProblem()
    file_path = 'ex3.dat'
    with open(file_path) as f:
        fp.load(f.readlines())

    reso = fp.solve()
    cost = fp.cost(reso)
    print(reso)
    print(cost)
