import math
import numpy as np


class FleetProblem(object):
    def __init__(self):
        self.fh = None
        self.sol = None
        self.P = None
        self.R = None
        self.td = None
        self.t = None
        self.Tod = None
        self.matriz = None
        self.V = None

    def load(self, fh):
        current_mode = None
        i = 0
        with open(fh, 'r') as f:
            for line in f:
                words = line.split()
                if words[0] == 'P':
                    current_line = 0
                    self.P = int(words[1])
                    self.matriz = [[0 for _ in range(self.P)] for _ in range(self.P)]
                    current_mode = 'P'
                    print(self.P)
                elif words[0] == 'R':
                    self.R = int(words[1])
                    current_mode = 'R'
                elif words[0] == 'V':
                    self.V = int(words[1])
                    current_mode = 'V'
                else:
                    if current_mode == 'P':
                        self.matriz[i] = words
                        i += 1
                    elif current_mode == 'R':
                        self.t.append([int(words[0]), int(words[1])])
                    elif current_mode == 'V':
                        self.Tod.append([int(words[0]), int(words[1])])
                    else:
                        raise Exception('Invalid mode')

            print(self.matriz)
        pass

    def cost(self, sol):

        pass


if __name__ == '__main__':
    fp = FleetProblem()
    fp.load('test.txt')
