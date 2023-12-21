import numpy as np
import random
import sys
import signal
import time
import pygame
import math
from copy import deepcopy


class connect4Player(object):
    def __init__(self, position, seed=0):
        self.position = position
        self.opponent = None
        self.seed = seed
        random.seed(seed)

    def play(self, env, move):
        move = [-1]


class human(connect4Player):

    def play(self, env, move):
        move[:] = [int(input('Select next move: '))]
        while True:
            if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
                break
            move[:] = [int(input('Index invalid. Select next move: '))]


class human2(connect4Player):

    def play(self, env, move):
        done = False
        while (not done):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.position == 1:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    move[:] = [col]
                    done = True


class randomAI(connect4Player):

    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        move[:] = [random.choice(indices)]


class stupidAI(connect4Player):
    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        if 3 in indices:
            move[:] = [3]
        elif 2 in indices:
            move[:] = [2]
        elif 1 in indices:
            move[:] = [1]
        elif 5 in indices:
            move[:] = [5]
        elif 6 in indices:
            move[:] = [6]
        else:
            move[:] = [0]


class minimaxAI(connect4Player):

    def simulateMove(self, env, move, player):
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[0].append(move)

    def play(self, env, move):
        depth = 3  # set depth
        possible = env.topPosition >= 0  # possible positions consist of wherever env.topPosition is greater than 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        v = -np.inf  # set value to -inf
        for column in indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, self.position)
            new_value = self.MIN(newState, depth - 1)
            if new_value > v:
                v = new_value
                move[:] = [column]
        print("Finished in time")

    def MAX(self, env, depth):
        if env.gameOver(env.history[0][-1], self.position):
            return -1
        if depth == 0:
            # print("eval: ", self.eval(env))
            return self.eval(env)

        v = -np.inf
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        for column in indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, self.position)
            v = max(v, self.MIN(newState, depth - 1))
        print("eval: ", self.eval(env))
        return v

    def MIN(self, env, depth):
        oppPlayer = {1: 2, 2: 1}
        if env.gameOver(env.history[0][-1], oppPlayer[self.position]):  # if the game is over for opposing player
            return 1
        if depth == 0:
            # print("eval: ", self.eval(env))
            return self.eval(env)

        v = np.inf
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        for column in indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, oppPlayer[self.position])
            v = min(v, self.MAX(newState, depth - 1))
        print("eval: ", self.eval(env))
        return v

    def eval(self, env):
        player_pos = (env.board == self.position).astype(
            int)  # a matrix where 1s are present in the player's token positions by comparing each board position to the player position and outputting a boolean value
        opp_pos = (env.board == 3 - self.position).astype(
            int)  # a matrix where 1s are present in the opposition's token positions by comparing each board position to the player position and outputting a boolean value
        weighted_board = player_pos - opp_pos  # subtract opposition's matrix from player's matrix to get a 1 in the player positions and -1 in the opposition's positions
        weights = np.array([[3, 4, 5, 7, 5, 4, 3],
                            [4, 6, 8, 10, 8, 6, 4],
                            [5, 8, 10, 13, 10, 8, 5],
                            [5, 8, 10, 13, 10, 8, 5],
                            [7, 8, 11, 12, 11, 8, 7],
                            [8, 11, 11, 13, 11, 11, 8]])

        weighted_matrix = np.multiply(weighted_board,
                                      weights)  # multiply the board of -1s and 1s by the weighted matrix
        return np.sum(weighted_matrix)  # sum up matrix values to get final evaluation function value

    def signal_handler(self):
        print("SIGTERM ENCOUNTERED")
        sys.exit(0)


class alphaBetaAI(connect4Player):

    def simulateMove(self, env, move, player):
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[0].append(move)

    def play(self, env, move):
        depth = 3  # set depth
        possible = env.topPosition >= 0  # possible positions consist of wherever env.topPosition is greater than 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        v = -np.inf  # set value to -inf
        alpha = -np.inf
        beta = np.inf

        ordered_indices = self.successor(env, indices)

        for column in ordered_indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, self.position)
            new_value = self.MIN(newState, depth - 1, alpha, beta)
            if new_value > v:
                v = new_value
                move[:] = [column]
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        print("Finished in time")

    def MAX(self, env, depth, alpha, beta):
        if env.gameOver(env.history[0][-1], self.opponent.position):
            return -123457
        if depth == 0:
            # print("eval: ", self.eval(env))
            return self.eval(env)

        v = -np.inf
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)

        ordered_indices = self.successor(env, indices)
        for column in ordered_indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, self.position)
            v = max(v, self.MIN(newState, depth - 1, alpha, beta))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        # print("eval: ", self.eval(env))
        return v

    def MIN(self, env, depth, alpha, beta):
        if env.gameOver(env.history[0][-1], self.position):  # if the game is over for opposing player
            return 123457
        if depth == 0:
            # print("eval: ", self.eval(env))
            return self.eval(env)

        v = np.inf
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        ordered_indices = self.successor(env, indices)
        for column in ordered_indices:
            newState = deepcopy(env)
            newState.visualize = False
            self.simulateMove(newState, column, self.opponent.position)
            v = min(v, self.MAX(newState, depth - 1, alpha, beta))
            beta = min(beta, v)
            if alpha >= beta:
                break
        # print("eval: ", self.eval(env))
        return v

    def eval(self, env):
        player_pos = (env.board == self.position).astype(int)  # a matrix where 1s are present in the player's token positions by comparing each board position to the player position and outputting a boolean value
        opp_pos = (env.board == 3 - self.position).astype(int)  # a matrix where 1s are present in the opposition's token positions by comparing each board position to the player position and outputting a boolean value
        weighted_board = player_pos - opp_pos  # subtract opposition's matrix from player's matrix to get a 1 in the player positions and -1 in the opposition's positions
        weights = np.array([[3, 4, 5, 7, 5, 4, 3],
                            [4, 6, 8, 11, 8, 6, 4],
                            [5, 8, 11, 13, 11, 8, 5],
                            [5, 8, 11, 13, 11, 8, 5],
                            [7, 10, 11, 12, 11, 8, 7],
                            [8, 10, 11, 13, 11, 10, 8]])

        weighted_matrix = np.multiply(weighted_board,
                                      weights)  # multiply the board of -1s and 1s by the weighted matrix
        # Counting combinations of 1 in a row, 2 in a row, and 3 in a row, vertical or diagonal

        num_consecutive = 0
        for row in range(6):
            for col in range(7):
                if env.board[row][col] == self.opponent.position:
                    # Check horizontal
                    if col < 7 - 3:
                        if env.board[row][col + 1] == self.position and env.board[row][col + 2] == self.position:
                            num_consecutive += 3
                    # Check vertical
                    if row < 6 - 3:
                        if env.board[row + 1][col] == self.position and env.board[row + 2][col] == self.position:
                            num_consecutive += 3
                    # Check diagonal
                    if row < 6 - 3 and col < 7 - 3:
                        if env.board[row + 1][col + 1] == self.position and env.board[row + 2][col + 2] == self.position:
                            num_consecutive += 3
                    if row < 6 - 3 and col >= 3:
                        if env.board[row + 1][col - 1] == self.position and env.board[row + 2][col - 2] == self.position:
                            num_consecutive += 3

        return np.sum(weighted_matrix) + num_consecutive

    def successor(self, env, indices):
        # Shuffle the indices randomly
        random.shuffle(indices)
        # Return the shuffled list of indices
        return indices

        """
        values = [(column, self.eval(env)) for column in indices]
        # Sort the list of tuples in descending order of evaluation function value
        values.sort(key=lambda x: x[1], reverse=True)
        # Return a list of column indices sorted by descending evaluation function value
        return [value[0] for value in values]
        """


SQUARESIZE = 100
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
