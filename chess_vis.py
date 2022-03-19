# chess_vis.py
# define a series of functions to facilitate visualization
import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *

# define a progress odometer
def print_progress(n, N): # n - the number of episodes performed so far; N - total number of episodes
    print (f'Completed training: {100 * n / N:.2f}%', end = '\r') if n != None else print (f'Completed training: {100}% ------------', end = '\r')


def exp_ma(arr, window, x_label = 'Training Time', y_label = 'Reward / Moves per episode', title = None, ax = None, color = None, label = None): # exponential moving average, taking an array and a window length as input

    # Exponential Moving Average (EMA) calculation

    alpha = 2 / (window + 1) # calculate the weighted multiplier

    ema = [arr[0]] # initiate the ema result vector by assigning the first element of arr to it

    for i in range(1, len(arr)): # traverse all elements from the second
        ema.append(alpha * arr[i] + (1 - alpha) * ema[-1]) # DP style implementation: s_t = alpha * y_t + (1 - alpha) * s_{t-1}


    # Result visualization
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 4), constrained_layout = True) # instantiate a subplots object
    ax.plot(ema, color = color, label = label) # plot the derived ema
    ax.set_xlabel(x_label) # format label of x axis
    ax.set_ylabel(y_label) # format label of y axis
    if title != None:
        ax.set_title(title) # format title


# define a function to translate piece location from board to the environment index system: a0 -> [3, 0]
def trans(pos):
    y = ord(pos[0]) - 97 # char c -> c* in ASCII -> c* - 97
    x = 4 - int(pos[1]) # char n -> int n - 1
    return (x, y)

# define a function to initiate the game lie the demo case
def demo_initialise_game(env, k1, q1, k2):
    # START THE GAME BY SETTING PIECIES

    size_board = 4; # 4-by-4 board
        
    env.Board = np.zeros([size_board, size_board], dtype=int) # initialize the board given size w/ a numpy zero matrix
    
    k1_x, k1_y = k1 # read the x,y coordinates of my king
    env.p_k1 = np.array([k1_x, k1_y]) # 1 for my king, pass its initial position to the env.
    env.Board[k1_x, k1_y] = 1 # update env.board w/ king's position
    
    q1_x, q1_y = q1 # read the x,y coordinates of my quen
    env.p_q1 = np.array([q1_x, q1_y]) # 2 for my queen, pass its initial position to the env.
    env.Board[q1_x, q1_y] = 2 # update env.board w/ queen's position

    k2_x, k2_y = k2 # read the x,y coordinates of opponent king
    env.p_k2 = np.array([k2_x, k2_y]) # 3 for oppenent king, pass its initial position to the env.
    env.Board[k2_x, k2_y] = 3 # update env.board w/ opponent king's position

    # Allowed actions for the agent's king
    env.dfk1_constrain, env.a_k1, env.dfk1 = degree_freedom_king1(env.p_k1, env.p_k2, env.p_q1, env.Board)
    
    # Allowed actions for the agent's queen
    env.dfq1_constrain, env.a_q1, env.dfq1  = degree_freedom_queen(env.p_k1, env.p_k2, env.p_q1, env.Board)
    
    # Allowed actions for the enemy's king
    env.dfk2_constrain, env.a_k2, env.check = degree_freedom_king2(env.dfk1, env.p_k2, env.dfq1, env.Board, env.p_k1)
    
    # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
    allowed_a=np.concatenate([env.a_q1,env.a_k1],0)
    
    # FEATURES (INPUT TO NN) AT THIS POSITION
    X=env.Features()

    return env.Board, X, allowed_a


# define a function to print the chess board w/ given position of pieces
def print_board(S, num = None, figsize = (5, 5), ax = None):

        # instantiate a subplots obj w/ default figsize = (5, 5)
        if ax == None:
                fig, ax = plt.subplots(figsize = figsize)
        
        # create a blank board
        blank_board = np.add.outer(range(4), range(4))%2 # create grids
        ax.imshow(blank_board, cmap="binary_r") # use 2D raster to generate the board
        ax.set_xticks([0.0, 1.0, 2.0, 3.0]) # formulate x ticks
        ax.set_xticklabels(['a', 'b', 'c', 'd']) # formulate x ticks
        ax.set_yticks([0.0, 1.0, 2.0, 3.0]) # formulate y ticks
        ax.set_yticklabels([4, 3, 2, 1]) # formulate y ticks


        # put pieces on the board
        board_idx = np.array([[(x, y) for y in range(4)] for x in range(4)]) # generate a positional index table
        pieces = ['K', 'Q', 'K'] # piece names
        colors = ['blue', 'blue', 'red'] # piece colors (blue -> player, red -> opponent)
        for i in range(3): # assign each piece
                y_loc, x_loc = board_idx[S == i + 1][0]
                ax.text(x_loc, y_loc, pieces[i], style='italic', color = colors[i],
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # format title
        if num == None:
                ax.set_title(f'Overkilling Endgame')         
        else:
                ax.set_title(f'Move: {num}') 

        # show figure
        if ax == None:
                fig.show()
