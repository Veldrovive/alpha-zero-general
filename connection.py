from typing import List

from MCTS import MCTS
from gomaku.GomakuGame import GomakuGame
from gomaku.pytorch.NNet import NNetWrapper as NNet
from utils import *

import numpy as np

inverse_content = {
    "b": -1,
    "-": 0,
    "w": 1
}

game = GomakuGame(8)

class CompetitionPlayer:
    def __init__(self, net_args: dotdict, checkpoint="./checkpoints/best.pth.tar", iterations: int=50):
        self.checkpoint = checkpoint
        self.size = game.getBoardSize()[0]
        self.mcts_args = dotdict({'numMCTSSims': iterations, 'cpuct': 1.0})
        self.net_args = net_args
        self.net = NNet(game, self.net_args)
        self.mcts = MCTS(game, self.net, self.mcts_args)

    def from_string_array(self, board_seed):
        board = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                val = inverse_content[board_seed[y][x]]
                board[y, x] = val
        return board

    def action_to_yx(self, action):
        action_y, action_x = action // self.size, action % self.size
        return action_y, action_x

    def yx_to_action(self, y, x):
        return y * self.size + x

    def move(self, board, player):
        player = 1 if player == "w" else -1
        board = self.from_string_array(board)

        canonical_board = game.getCanonicalForm(board, player)
        action_probs = self.mcts.getActionProb(canonical_board, temp=0)
        action = np.argmax(action_probs)
        valid_actions = game.getValidMoves(canonical_board, 1)

        if valid_actions[action] == 0:
            # Then the agent tried to play an invalid move.
            # Mask the probs with the valid actions and select from that.
            valid_action_probs = action_probs*valid_actions
            action = np.argmax(valid_action_probs)

        return self.action_to_yx(action)


class Player:
    def __init__(self, checkpoint = "./checkpoints/best.pth.tar", iterations=50):
        self.checkpoint = checkpoint
        self.size = 8
        self.net = NNet(game)
        self.args = dotdict({'numMCTSSims': iterations, 'cpuct': 1.0})
        self.mcts = MCTS(game, self.net, self.args)

    def from_string_array(self, board_seed: List[List[str]]):
        # This is the format the game will be delivered in in the competition
        board = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                val = inverse_content[board_seed[y][x]]
                board[y, x] = val
        return board

    def action_to_yx(self, action):
        action_y, action_x = action // self.size, action % self.size
        return action_y, action_x

    def yx_to_action(self, y, x):
        return y * self.size + x

    def move(self, board, player, tournament_style=False):
        if tournament_style:
            player = 1 if player == "w" else -1
            board = self.from_string_array(board)

        canonical_board = game.getCanonicalForm(board, player)
        action_probs = self.mcts.getActionProb(canonical_board, temp=0)
        action = np.argmax(action_probs)
        valid_actions = game.getValidMoves(canonical_board, 1)

        if valid_actions[action] == 0:
            # Then the agent tried to play an invalid move.
            # Mask the probs with the valid actions and select from that.
            valid_action_probs = action_probs*valid_actions
            action = np.argmax(valid_action_probs)

        return self.action_to_yx(action)
