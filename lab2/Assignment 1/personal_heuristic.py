from heuristic import Heuristic
from make_grid import SIZE

class PersonalH(Heuristic):

    @staticmethod
    def h(node):
        a = (SIZE-int(node.i))
        b = (SIZE-int(node.j))
        return max(a, b)