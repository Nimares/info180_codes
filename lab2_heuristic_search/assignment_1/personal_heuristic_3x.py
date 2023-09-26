from heuristic import Heuristic
from make_grid import SIZE

class PersonalH3x(Heuristic):

    @staticmethod
    def h(node):
        a = (SIZE-int(node.i))
        b = (SIZE-int(node.j))
        longest_distance_from_goal = max(a, b)
        longest_distance_from_goal_3x = longest_distance_from_goal*3
        return longest_distance_from_goal_3x