class FleetProblem(object):

    def __init__(self):
        self.sol = None
        self.P = None
        self.R = None
        self.V = None
        self.graph = []
        self.requests = []
        self.vehicles = []

    def load(self, fh):
        current_mode = None
        with open(fh, 'r') as f:
            for line in f:
                words = line.split()

                if not words:
                    continue  # Skip empty lines

                if words[0] == 'P':
                    self.P = int(words[1])
                    current_mode = 'P'

                elif words[0] == 'R':
                    self.R = int(words[1])
                    current_mode = 'R'

                elif words[0] == 'V':
                    self.V = int(words[1])
                    current_mode = 'V'

                else:
                    if current_mode == 'P':
                        row = list(map(int, words))
                        self.graph.append(row)
                    elif current_mode == 'R':
                        request = list(map(int, words))
                        self.requests.append(request)
                    elif current_mode == 'V':
                        vehicle = list(map(int, words))
                        self.vehicles.append(vehicle)
                    else:
                        raise Exception('Invalid mode')
        print(self.graph)
        print(self.requests)
        pass

    def cost(self, sol):
        td = 0
        t = 0
        tod = 0
        final_cost = 0
        for action in sol:
            if action[0] == 'Dropoff':
                td = action[3]
                t = self.requests[action[2] - 1][0]
                tod = "?"

            final_cost += td - t - tod
        print(final_cost)
        pass


if __name__ == '__main__':
    sols = [('Pickup', 0, 3, 30.0), ('Pickup', 1, 4, 25.0), ('Pickup', 1, 0, 25.0),
           ('Dropoff', 1, 4, 75.0), ('Dropoff', 1, 0, 75.0), ('Pickup', 0, 2, 30.0),
           ('Dropoff', 0, 3, 80.0), ('Pickup', 0, 1, 80.0), ('Dropoff', 0, 1, 140.0),
           ('Dropoff', 0, 2, 140.0)]

    fp = FleetProblem()
    file_path = 'test.txt'
    fp.load(file_path)
    fp.cost(sols)
