from heuristic import Heuristic
import make_grid

class Ass3(Heuristic):

    @staticmethod
    def h(node):
        grid = make_grid.SIZE
        return 0