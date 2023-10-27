#!/usr/bin/env python3

from search import Problem, astar_search
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

        sorted_fleet = sorted(self.fleet, key=lambda x: x.get_capacity(), reverse=True)
        sorted_requests = sorted(self.requests, key=lambda x: x.get_passengers(), reverse=True)

        self.requests = tuple(sorted_requests)
        self.fleet = tuple(sorted_fleet[:request_index])

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

        # Iterate over each request
        for request in requests:
            # Check if the request is waiting
            if request.get_status() == 'waiting':
                # Iterate over each vehicle
                for vehicle in vehicles:
                    # Check if the vehicle has enough capacity for the request passengers
                    if vehicle.get_capacity() >= request.get_passengers():
                        # Get the tiem of the veihicle
                        time = vehicle.get_time()

                        # Calculate the time it takes for the vehicle to reach the pickup location
                        travel_time = self.graph.get_edge(vehicle.get_location(), request.get_pickup())
                        arrival_time = time + travel_time
                        request_time = request.get_time()
                        pickup_time = max(arrival_time, request_time)

                        # Add a 'Pickup' action to the list of actions
                        actions.append(Action('Pickup', vehicle.get_index(), request.get_index(), pickup_time))

            # Check if the request is traveling
            elif request.get_status() == 'traveling':
                # Get the vehicle assigned to the request
                vehicle = state.get_vehicle(request.get_vehicle_id())

                # Get the time of the vehicle
                time = vehicle.get_time()

                # Calculate the time it takes for the vehicle to reach the dropoff location
                travel_time = self.graph.get_edge(vehicle.get_location(), request.get_dropoff())
                dropoff_time = time + travel_time

                # Add a 'Dropoff' action to the list of actions
                actions.append(Action('Dropoff', vehicle.get_index(), request.get_index(), dropoff_time))

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
        vehicle = new_state.get_vehicle(action.get_vehicle_id())
        request = new_state.get_request(action.get_request_id())

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
        """Returns the cost of a solution.

        Args:
            c (float): The cost of the path to the node.
            state1 (State): The state of the node.
            action (Action): The action to execute.
            state2 (State): The state of the child node.

        Returns:
            float: The cost of the solution.
        """
        # Get the request from the first state using the IDs from the action
        request = state1.get_request(action.get_request_id())

        # Get the time from the action
        time = action.get_time()

        # If the action is a 'Pickup' action
        if action.get_type() == 'Pickup':
            # Add to the cost the difference between the time of the action and the time of the request
            c += (time - request.get_time())
        else:
            # If it's not a 'Pickup' action, it's a 'Dropoff' action. Subtract from the cost
            # the difference between the time of the action and
            # the time it takes for the vehicle to go from its location to the dropoff location,
            # and subtract also the status time of the request
            c += time - request.get_status_time() - self.graph.get_edge(request.get_pickup(), request.get_dropoff())

        return c  # Return the cost

    def h(self, node):
        """
        This function calculates the heuristic value for a given node in a problem.
        The heuristic value is a measure of the cost to reach the goal from the given node.

        Args:
            node (Node): The node to calculate the heuristic value for.

        Returns:
            float: The calculated heuristic value.
        """

        # Get the state from the node
        state = node.state

        # Get the requests and vehicles from the state
        requests = state.get_requests()
        vehicles = state.get_vehicles()

        if len(requests) == len(vehicles):
            # If there are more vehicles than requests, use the heuristic for lots of vehicles
            return self.calculate_heuristic_lots_of_vehicles(vehicles, requests)
        # If there are more requests than vehicles, use the heuristic for fewer vehicles
        return self.calculate_heuristic_fewer_vehicles(vehicles, requests, state)

    def calculate_heuristic_lots_of_vehicles(self, vehicles, requests):
        '''This function calculates the heuristic value for a given node in a problem.
            When there are more vehicles than requests, the heuristic value is a measure of what requests are
            assigned to what vehicles, since is more worth to assign only one request to each vehicle

            Args:
                vehicles (list): The vehicles in the state.
                requests (list): The requests in the state.

            Returns:
                float: The calculated heuristic value.
        '''
        pre_notion = []
        index = 0
        h = 0
        # assign a request to a vehicle
        for request in self.requests:
            if request.get_passengers() <= self.fleet[index].get_capacity():
                pre_notion.append(self.fleet[index].get_index())
            index += 1

        index = 0
        for request in requests:
            if request.get_status() == 'traveling' and request.get_vehicle_id() != pre_notion[index]:
                # If a request is traveling, and it's not assigned to the vehicle in the solution,
                # add infinity to the heuristic
                h += float('inf')
            if request.get_status() == 'waiting':
                # If a request is waiting, add 10 to the heuristic
                h += 10
            index += 1

        return h

    def calculate_heuristic_fewer_vehicles(self, vehicles, requests, state):
        '''This function calculates the heuristic value for a given node in a problem.
            When there are more requests than vehicles, the heuristic value is a measure of the cost to reach the goal

            Args:
                vehicles (list): The vehicles in the state.
                requests (list): The requests in the state.
                state (State): The state to calculate the heuristic value for.

            Returns:
                float: The calculated heuristic value.
        '''
        h = 0

        # Iterate over each request
        for request in requests:
            # Check if the request is waiting
            if request.get_status() == 'waiting':
                # If a request is waiting, find the minimum cost of picking up the request
                min_cost = float('inf')
                # Iterate over each vehicle
                for vehicle in vehicles:
                    cost = float('inf')

                    # Check if the vehicle has enough capacity for the request
                    if vehicle.get_capacity() >= request.get_passengers():
                        # Calculate the cost of picking up the request
                        cost = self.pickup_delay(vehicle, request)
                    else:
                        # If a vehicle doesn't have enough capacity for a request,
                        # calculate the cost of dropping off other requests first
                        vehicle_requests = vehicle.get_requests()
                        for index in vehicle_requests:
                            vehicle_request = state.get_request(index)
                            # if vehicle.get_capacity() + vehicle_request.get_passengers() >= request.get_passengers():
                            time = vehicle.get_time()

                            travel_time = (self.graph.get_edge(vehicle.get_location(), vehicle_request.get_dropoff()) + 
                                            self.graph.get_edge(vehicle_request.get_dropoff(), request.get_pickup()))
                            arrival_time = time + travel_time
                            request_time = request.get_time()
                            pickup_time = max(arrival_time, request_time)
                            cost = pickup_time - request_time
                    if cost < min_cost:
                        min_cost = cost

                # Add the minimum cost to the heuristic
                h += min_cost
            
            # Check if the request is traveling
            elif request.get_status() == 'traveling':
                # Get the vehicle assigned to the request
                vehicle = state.get_vehicle(request.get_vehicle_id())

                # Calculate the cost of dropping off the request
                h += self.dropoff_delay(vehicle, request)

        return h

    def pickup_delay(self, vehicle, request):
        """Calculates the cost of picking up a request.

            Args:
                vehicle (Vehicle): The vehicle to pick up the request.
                request (Request): The request to pick up.

            Returns:
                float: The calculated cost of picking up the request.
        """
        time = vehicle.get_time()

        travel_time = self.graph.get_edge(vehicle.get_location(), request.get_pickup())
        arrival_time = time + travel_time
        request_time = request.get_time()
        pickup_time = max(arrival_time, request_time)

        return pickup_time - request_time
    
    def dropoff_delay(self, vehicle, request):
        """Calculates the cost of dropping off a request.

            Args:
                vehicle (Vehicle): The vehicle to drop off the request.
                request (Request): The request to drop off.

            Returns:
                float: The calculated cost of dropping off the request.
        """
        time = vehicle.get_time()

        travel_time = self.graph.get_edge(vehicle.get_location(), request.get_dropoff())
        dropoff_time = time + travel_time

        return dropoff_time - request.get_status_time() - self.graph.get_edge(request.get_pickup(), request.get_dropoff())

    def solve(self):
        """Solves the problem using A star search."""
        result = astar_search(self, display=True)
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
                and self.status_time == other.status_time
                and self.index == other.index
                and self.vehicle_id == other.vehicle_id
        )

    def get_index(self):
        """Gets the index of the request."""
        return self.index

    def __lt__(self, other):
        """Checks if the request is less than another request."""
        return self.status_time < other.status_time

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
    
    def set_status_time(self, time):
        """Sets the status time of the request."""
        self.status_time = time

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
    
    def get_true_capacity(self):
        """Gets the capacity of the vehicle."""
        return self.capacity

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
        score1 = 0
        score2 = 0

        for i in range(len(self.requests)):
            if self.requests[i].get_status() == 'completed':
                score1 += 3
            if self.requests[i].get_status() == 'traveling':
                score1 += 2
            if self.requests[i].get_status() == 'waiting':
                score1 += 1

            if other.requests[i].get_status() == 'completed':
                score2 += 3
            if other.requests[i].get_status() == 'traveling':
                score2 += 2
            if other.requests[i].get_status() == 'waiting':
                score2 += 1

        return score1 < score2

    def __hash__(self):
        """ The hash value of the requests."""
        return self.requests.__hash__()

    def __eq__(self, other):
        """Checks if two states are equal."""
        return self.requests == other.requests

    def get_requests(self):
        """Gets the requests in the state."""
        return self.requests
    
    def get_request(self, request_id):
        """Gets a request in the state."""
        for request in self.requests:
            if request.get_index() == request_id:
                return request

    def get_vehicles(self):
        """Gets the vehicles in the state."""
        return self.vehicles

    def get_vehicle(self, vehicle_id):
        """Gets a vehicle in the state."""
        for vehicle in self.vehicles:
            if vehicle.get_index() == vehicle_id:
                return vehicle


if __name__ == '__main__':
    fp = FleetProblem()
    file_path = 'tests/ex9.dat'
    with open(file_path) as f:
        fp.load(f.readlines())

    reso = fp.solve()
    cost = fp.cost(reso)
    print(reso)
    print(cost)
