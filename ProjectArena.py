from connection import Player
from gomaku.GomakuGame import GomakuGame
from time import time
import numpy as np
import torch
import random

def print_board(game, board):
    str_board = game.stringRepresentation(board)
    print(str_board)


def play_gomoku_auto(player_1: Player, player_2: Player, r_seed=None, verbose=False):
    if r_seed is not None:
        np.random.seed(r_seed)
        torch.manual_seed(r_seed)
        random.seed(r_seed)

    game = GomakuGame(8)
    board = game.getInitBoard(play_random_moves=2)

    if verbose:
        print("Initial Board:")
        print_board(game, board)

    # log()
    p1_times = []
    p2_times = []

    def get_winner(game_res):
        if abs(game_res) < 1:
            return None
        return game_res

    while True:
        s = time()
        move_y, move_x = player_1.move(board, 1, tournament_style=False)
        p1_times.append(time() - s)
        if verbose:
            print(f"\n\n**************\nWhite move:({move_y}, {move_x})")
        board = game.getNextState(board, 1, player_1.yx_to_action(move_y, move_x))[0]

        if verbose:
            print_board(game, board)
        game_res = game.getGameEnded(board, 1)
        if game_res != 0:
            winner = get_winner(game_res)
            if verbose:
                if winner is None:
                    print("Draw")
                print("White WON!!" if game_res == 1 else "Black WON!!")
            return winner, p1_times, p2_times

        s = time()
        move_y, move_x = player_2.move(board, -1, tournament_style=False)
        p2_times.append(time() - s)
        if verbose:
            print(f"\n\n**************\nBlack move:({move_y}, {move_x})")
        board = game.getNextState(board, -1, player_2.yx_to_action(move_y, move_x))[0]

        if verbose:
            print_board(game, board)
        game_res = game.getGameEnded(board, 1)
        if game_res != 0:
            winner = get_winner(game_res)
            if verbose:
                if winner is None:
                    print("Draw")
                print("White WON!!" if game_res == 1 else "Black WON!!")
            return winner, p1_times, p2_times

def get_stats(times, verbose=False):
    avg, std = np.average(times), np.std(times)
    if verbose:
        print(f"Average time: {avg} and std: {std}")
    return avg, std


if __name__ == "__main__":
    player_1 = Player(50)
    player_2 = Player("checkpoints/checkpoint_20.pth.tar", 50)
    _, p1_times, p2_times = play_gomoku_auto(player_1, player_2, r_seed=None, verbose=True)
    print("Player 1:")
    get_stats(p1_times, True)
    print("Player 2:")
    get_stats(p2_times, True)
