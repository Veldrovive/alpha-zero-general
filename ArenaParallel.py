import logging
from utils import dotdict
from gomaku.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
import numpy as np
import concurrent.futures as conc
from time import sleep
from gomaku.GomakuGame import GomakuGame
import os

try:
    if 'google.colab' in str(get_ipython()):
      from tqdm.notebook import tqdm
    else:
      from tqdm import tqdm
except NameError:
    from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1_checkpoint, player2_checkpoint, num_agents, game, net_1_args=None, net_2_args=None, args=None, display=None, names=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1_checkpoint
        self.player2 = player2_checkpoint
        self.num_agents = num_agents
        self.game = game
        self.display = display
        self.args = args or dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        self.names = names or ("Player_1", "Player_2")

        # If player1 or player2 are checkpoints, load them. Otherwise the same agent will play for each.
        if isinstance(self.player1, str):
            self.player1s = [self.load_agent(self.player1, net_1_args) for _ in range(self.num_agents)]
        else:
            self.player1s = [self.player1 for _ in range(self.num_agents)]
        if isinstance(self.player2, str):
            self.player2s = [self.load_agent(self.player2, net_2_args) for _ in range(self.num_agents)]
        else:
            self.player2s = [self.player2 for _ in range(self.num_agents)]
        self.to_play = 0
        self.left_to_play = 0

    def load_agent(self, checkpoint, args):
        net = NNet(self.game, net_args=args)
        net.load_checkpoint(filename=checkpoint)
        mcts = MCTS(self.game, net, self.args)
        return lambda board: np.argmax(mcts.getActionProb(board, temp=0))

    def playGame(self, agent_index, verbose=False, reverse=False, save_index=-1):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        file = None
        if save_index > -1:
            filename = f"./tmp/{self.names[0]}_vs_{self.names[1]}_match_{save_index}"
            os.makedirs("./tmp", exist_ok=True)
            file = open(f"{filename}.txt", "w")

        def log_move(player, action, board):
            if save_index > -1:
                if player == 1:
                    color = "Black"
                else:
                    color = "White"
                if reverse:
                    player *= -1
                player_index = int((player + 1) / 2)
                player_name = self.names[player_index]
                str_board = self.game.stringRepresentation(board, highlight_action=action, include_numbers=True)
                y, x = action // 8, action % 8
                file.write("********************\n")
                file.write(f"Player {player_name} as {color} played ({y}, {x})\n")
                file.write(str_board)
                file.write("\n\n")


        if reverse:
            players = [self.player1s[agent_index], None, self.player2s[agent_index]]
        else:
            players = [self.player2s[agent_index], None, self.player1s[agent_index]]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            log_move(curPlayer, action, board)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        winner = (-1 if reverse else 1) * curPlayer * self.game.getGameEnded(board, curPlayer)
        try:
            file.close()
            if abs(winner) < 1:
                os.rename(f"{filename}.txt", f"{filename}_draw.txt")
            else:
                w = 1-int((winner+1)/2)
                os.rename(f"{filename}.txt", f"{filename}_winner_{self.names[w]}.txt")
        except AttributeError:
            pass
        return winner

    def handle_agent(self, agent_index, results, log=False):
        """
        Plays games with agents[agent_index] storing winner in results until self.to_play == 0
        :param agent_index:
        :param results:
        :return:
        """
        while self.left_to_play > 0:
            self.left_to_play -= 1  # TODO: Figure out if this is a race condition. Since this is threading and not multiprocessesing I don't see why it would be.
            reverse = self.left_to_play < self.to_play/2
            results.append(self.playGame(agent_index, save_index=self.to_play-self.left_to_play if log else -1, reverse=reverse))
            self.played += 1

    def playGamesParallel(self, num, verbose=False, log=False):
        """
        Plays num games with a thread for each agent.
        :param num:
        :param agent_index:
        :param verbose:
        :return:
        """
        self.to_play = num
        self.left_to_play = num
        self.played = 0
        results = []
        with conc.ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            executor.map(lambda p: self.handle_agent(*p), [(i, results, log) for i in range(self.num_agents)])
            with tqdm(total=self.to_play, desc="Arena Games (Parallelized)") as pbar:
                last = self.played
                while self.played < self.to_play:
                    if last != self.played:
                        res = np.array(results)
                        total = len(results)
                        player_1_won = np.count_nonzero(res == 1)
                        player_2_won = np.count_nonzero(res == -1)
                        draws = total - player_1_won - player_2_won
                        pbar.set_description_str(f"1 Winning: {round(player_1_won*100/total, 2)}%. 2 Winning: {round(player_2_won*100/total, 2)}%. {round(draws*100/total, 2)}% Draws")
                        pbar.n = self.played
                        pbar.refresh()
                        last = self.played
                    sleep(0.1)
                pbar.n = self.to_play
                pbar.refresh()
        results = np.array(results)
        player_1_won = np.count_nonzero(results == 1)
        player_2_won = np.count_nonzero(results == -1)
        draws = self.to_play - player_1_won - player_2_won
        self.to_play = 0
        return player_1_won, player_2_won, draws

if __name__ == "__main__":
    game = GomakuGame(8)
    arena = Arena("/Users/aidandempster/Desktop/EngSci/ESC190/alphaZero/checkpoints/best.pth.tar", "/Users/aidandempster/Desktop/EngSci/ESC190/alphaZero/checkpoints/checkpoint_20.pth.tar", 2, game)
    arena.playGamesParallel(4, log=True)