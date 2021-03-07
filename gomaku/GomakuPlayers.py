import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:  # TODO: Why the hell did they do this this way? Fix this shit.
            a = np.random.randint(self.game.getActionSize())
        return a