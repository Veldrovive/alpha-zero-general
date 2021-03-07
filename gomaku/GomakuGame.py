import sys
sys.path.append('..')
from Game import Game
# from .OthelloLogic import Board
import numpy as np

class GomakuGame(Game):
    content = {
        -1: "b",
        0: "-",
        1: "w"
    }

    def __init__(self, n):
        super(Game, self).__init__()
        self.size = n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return np.zeros((self.size, self.size))

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.size, self.size

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.size ** 2

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        action_y, action_x = action // self.size, action % self.size
        new_board = board.copy()
        new_board[action_y, action_x] = player
        return new_board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        zeros = np.where(board == 0)
        valid_moves = zeros[0] * self.size + zeros[1]
        all_moves = np.zeros(self.getActionSize(), dtype=int)
        all_moves[valid_moves] = 1
        return all_moves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """

        # Returns -1 is black won and 1 if white won. 0 if there was a draw. None if the game is ongoing.
        # TODO: Check if there is a valid 5 in a row
        def detect_five(y_start, x_start, d_y, d_x):
            seq_length = None
            seq_player = None
            y_max = self.size - 1
            x_max = self.size - 1
            while 0 <= y_start <= y_max and 0 <= x_start <= x_max:
                cur_player = board[y_start, x_start]
                if cur_player == seq_player:
                    seq_length += 1
                else:
                    if seq_length == 5:
                        return seq_player
                    seq_length = None
                    seq_player = None
                    if cur_player != 0:
                        seq_length = 1
                        seq_player = cur_player
                y_start += d_y
                x_start += d_x
            return None

        # Check for a winner
        winner = None
        x_start = 0
        for y_start in range(len(board)):
            # Check directions (0, 1) and (1, 1)
            winner = winner or detect_five(y_start, x_start, 0, 1)
            winner = winner or detect_five(y_start, x_start, 1, 1)
        x_start = len(board[0]) - 1
        for y_start in range(len(board)):
            # Check direction (1, -1)
            winner = winner or detect_five(y_start, x_start, 1, -1)
        y_start = 0
        for x_start in range(len(board[0])):
            # Check direction (1, 0)
            winner = winner or detect_five(y_start, x_start, 1, 0)
            if x_start > 1:
                # Check the rows that were not on the y pass
                winner = winner or detect_five(y_start, x_start, 1, 1)
            if x_start < len(board[0]) - 1:
                # Chck the rows that were not on the second y pass
                winner = winner or detect_five(y_start, x_start, 1, -1)
        if winner is not None:
            return player*winner  # TODO: Make sure this actually verifies the winner

        if np.max(self.getValidMoves(board, -player)) > 0:
            return 0
        return 0.01

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board*player

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        augmented_boards = []
        # We reshape the policy into a board so that when we rotate or reflect it the policy changes correctly
        policy_board = np.reshape(pi, (self.size, self.size))

        for i in [0, 1, 2, 3]:  # For 0, 90, 180, and 270 degrees of rotation
            rot_board = np.rot90(board, i)
            rot_policy_board = np.rot90(policy_board, i)
            flipped_board = np.fliplr(rot_board)
            flipped_policy_board = np.fliplr(rot_policy_board)
            augmented_boards.extend(
                [(rot_board, rot_policy_board.ravel()), (flipped_board, flipped_policy_board.ravel())])
        return augmented_boards

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return '\n'.join([''.join([self.content[x] for x in line]) for line in board])

