class FleetProblem:
    def __init__(self):
        self.sol = None
        self.graph = None
        self.requests = None
        self.vehicles = None

    def load(self, fH):
        current_mode = None
        row = 0
        P = 0

        with open(fH, 'r') as f:
            for line in f:
                words = line.split()

                if not words or words[0].startswith('#'):
                    continue  # Skip empty lines and comments

                if words[0] == 'P':
                    P = int(words[1])
                    self.graph = Graph(P)
                    current_mode = 'P'
                elif words[0] == 'R':
                    self.requests = Requests(int(words[1]))
                    current_mode = 'R'
                elif words[0] == 'V':
                    self.vehicles = Vehicles(int(words[1]))
                    current_mode = 'V'
                else:
                    if current_mode == 'P':
                        col = 1 - P
                        for weight in words:
                            self.graph.add_edge(row, col, int (weight))
                            col += 1
                        row += 1
                        P -= 1
                    elif current_mode == 'R':
                        request = tuple(map(int, words))
                        self.requests.add_request(request)
                    elif current_mode == 'V':
                        vehicle = int(words[0])
                        self.vehicles.add_vehicle(vehicle)
                    else:
                        raise Exception('Invalid mode')
                    
        self.graph.print_graph()
        self.requests.print_requests()
        self.vehicles.print_vehicles()

    def cost(self, sol):
        total_cost = 0
        for action in sol:
            if action[0] == 'Dropoff':
                request = self.requests.get_request(action[2])

                td = action[3]
                t = request[0]
                Tod = self.graph.get_edge(request[1], request[2])

                total_cost += td - t - Tod

        return total_cost


class Graph:
    def __init__(self, num_vertices, directed=False):
        self.num_vertices = num_vertices
        self.directed = directed
        self.graph = [[0] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        if not self.directed:
            self.graph[v][u] = w

    def get_edge(self, u, v):
        return self.graph[u][v]
    
    def print_graph(self):
        print('Graph:')
        for row in self.graph:
            print(row)
        print()
    
class Requests:
    def __init__(self, num_requests):
        self.num_requests = num_requests
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)

    def get_request(self, index):
        return self.requests[index]
    
    def print_requests(self):
        print('Requests:')
        for request in self.requests:
            print(request)
        print()
    
class Vehicles:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.vehicles = []

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def get_vehicle(self, index):
        return self.vehicles[index]
    
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
    fp.load(file_path)
    print('Total delay:', fp.cost(sols))
