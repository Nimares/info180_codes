from heuristic import Heuristic
from make_grid import SIZE
import math

class EucledianH(Heuristic):

    @staticmethod
    def h(node):
        x1 = node.i
        y1 = node.j
        x2 = SIZE
        y2 = SIZE
        x_distance = x2 - x1
        y_distance = y2 - y1
        euclidian_distance = math.sqrt((x_distance)**2+(y_distance))

        euclidian_distance = math.sqrt((SIZE-node.i)*2+(SIZE-node.j)**2)
        return euclidian_distance
    
        