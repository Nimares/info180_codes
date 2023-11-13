'''
Class that represent the features and methods for an Improved  Computer Board game player
Improved evaluation function. Heuristics for number of pieces and postion of pieces. 
'''


from Board import CROSS, RING, EMPTY
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
        cross_step_from_victory = Board.GAMESIZE-1
        ring_step_from_victory = Board.GAMESIZE-1
        o_clear_path = False
        o_clear_path = False
        for i in range(Board.GAMESIZE):
            for j in range(Board.GAMESIZE):
                if a_board.the_grid[i][j] == CROSS:
                    cross_count += 1
                    if i > max_cross_row:
                        max_cross_row = i
                    for k in range(i+1, Board.GAMESIZE):
                        if a_board.the_grid[k][j] == EMPTY:
                            x_clear_path = True
                        else: 
                            x_clear_path = False
                    if x_clear_path:
                        if (Board.GAMESIZE-1)-(i) < cross_step_from_victory:
                            cross_step_from_victory = Board.GAMESIZE-(i)
                if a_board.the_grid[i][j] == RING:
                    ring_count += 1
                    if i < max_ring_row:
                        max_ring_row = i
                    for k in range(Board.GAMESIZE-1, i):
                        if a_board.the_grid[k][j] == EMPTY:
                            o_clear_path = True
                        else: 
                            o_clear_path = False
                    if o_clear_path:
                        if i < ring_step_from_victory:
                            ring_step_from_victory = (i)
        score = 0
        if self.mark == CROSS:
            score = (max_cross_row - (Board.GAMESIZE-1-max_ring_row))
            + (cross_count-ring_count)
            # + (ring_step_from_victory-cross_step_from_victory)
        if self.mark == RING:
            score = ((Board.GAMESIZE-1-max_ring_row) - max_cross_row) 
            + (ring_count-cross_count)
            # + (cross_step_from_victory-ring_step_from_victory)
        return score
