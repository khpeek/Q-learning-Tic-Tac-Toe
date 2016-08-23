import numpy as np
import Tkinter as tk
import copy
import cPickle as pickle    # cPickle is for Python 2.x only; in Python 3, simply "import pickle" and the accelerated version will be used automatically if available
from Q_Learning_Tic_Tac_Toe import Game, Board, Player, HumanPlayer, ComputerPlayer, RandomPlayer, THandPlayer, QPlayer     # Classes used for Tic Tac Toe

root = tk.Tk()
epsilon = 0.9
player1 = QPlayer(mark="X",epsilon = epsilon)
player2 = QPlayer(mark="O",epsilon = epsilon)
game = Game(root, player1, player2)

N_episodes = 200000
for episodes in range(N_episodes):
    game.play()
    game.reset()

Q = game.Q

filename = "Q_epsilon_09_Nepisodes_%s.p" % N_episodes
pickle.dump(Q, open(filename, "wb"))
