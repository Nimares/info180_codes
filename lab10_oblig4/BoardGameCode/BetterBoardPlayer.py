'''
Class that represent the features and methods for an Improved  Computer Board game player
Improved evaluation function. Heuristic now also takes consideration for moves that remove player pieces. 
'''


from Board import CROSS, RING
import Board
from BoardComputerPlayer import BoardComputerPlayer


class BetterBoardPlayer(BoardComputerPlayer):

    def __init__(self, the_mark):
        '''
        Constructor
        :param compatibility_score_set:
        '''
        super(BetterBoardPlayer, self).__init__(the_mark)
        self.name = "Better"

    def evaluate_game_status(self, a_board):
        max_cross_row = 0
        max_ring_row = Board.GAMESIZE-1
        cross_count = 0
        ring_count = 0
        for i in range(Board.GAMESIZE):
            for j in range(Board.GAMESIZE):
                if a_board.the_grid[i][j] == CROSS:
                    cross_count += 0
                    if i > max_cross_row:
                        max_cross_row = i
                if a_board.the_grid[i][j] == RING:
                    ring_count += 0
                    if i < max_ring_row:
                        max_ring_row = i
        score = 0
        if self.mark == CROSS:
            score = max_cross_row - (Board.GAMESIZE-1-max_ring_row) + (cross_count-ring_count)
        if self.mark == RING:
            score = (Board.GAMESIZE-1-max_ring_row) - max_cross_row + (ring_count-cross_count)
        return score
