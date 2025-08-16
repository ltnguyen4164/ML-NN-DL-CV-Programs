# Long Nguyen
# 1001705873

import numpy as np
import random

def value_iteration(data_file, ntr, gamma, K):
    # function that parses the environment file and returns the grid, dimensions, and terminal rewards
    def parse_file(data_file):
        grid = []
        with open(data_file, 'r') as fp:
            for line in fp:
                grid.append([float(x.strip()) if x.strip().lstrip('-').replace('.', '', 1).isdigit() else x.strip() for x in line.strip().split(',')])
        return grid

    # the set of actions available at each state s, A(s)
    actions = ["up", "down", "left", "right"] 

    # get environment
    grid = parse_file(data_file)
    print(grid)
    rows, cols = len(grid), len(grid[0])
    walls = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 'X'} 

    # initialize utility values
    U_prime = np.zeros((rows, cols))
    
    # reward function, R(s)
    def reward_func(row, col):
        if grid[row][col] == '.' or grid[row][col] == 'I':
            # non-terminal state
            return ntr
        elif grid[row][col] == 'X':
            # blocked state
            return 0
        else:
            # terminal state
            return float(grid[row][col])
        
    # transition model
    def transition_model(s, a):
        A = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),}
        perpendiculars = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"],}
        row, col = s
        probabilities = {}

        # helper function to determine if a cell is valid
        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and (r, c) not in walls and grid[r][c] != 'X'
        
        # add intended move
        dx, dy = A[a]
        next_row, next_col = row + dx, col + dy
        if is_valid(next_row, next_col):
            probabilities[(next_row, next_col)] = 0.8
        else:
            probabilities[(row, col)] = probabilities.get((row, col), 0) + 0.8

        # Add perpendicular moves
        for perp_action in perpendiculars[a]:
            dx, dy = A[perp_action]
            next_row, next_col = row + dx, col + dy
            if is_valid(next_row, next_col):
                probabilities[(next_row, next_col)] = probabilities.get((next_row, next_col), 0) + 0.1
            else:
                probabilities[(row, col)] = probabilities.get((row, col), 0) + 0.1
        return probabilities
    
    # main loop
    for k in range(K):
        U = np.copy(U_prime)
        # for each state, s
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 'X' and (grid[r][c] == '.' or grid[r][c] == 'I'):
                    # compute value for each action, max{∑[p(s'|s,a)U[s']]}
                    max_value = float('-inf')
                    for action in actions:
                        probs = transition_model((r, c), action)
                        value = sum(prob * U[next_r, next_c] for (next_r, next_c), prob in probs.items())
                        max_value = max(max_value, value)
                    # bellman update
                    U_prime[r, c] = reward_func(r, c) + gamma * max_value
                else:
                    U_prime[r, c] = reward_func(r, c)
    
    print("utilities:")
    for row in U_prime:
        print(" ".join(f"{value:6.3f}" for value in row))

    action_map = {'left': '<', 'right': '>', 'up': '^', 'down': 'v'}
    policy = policy_iteration(grid, actions, reward_func, transition_model, gamma, K)
    print("policy:")
    for row in range(len(grid)):
        line = ""
        for col in range(len(grid[0])):
            if grid[row][col] == 'X':
                # blocked
                line += f"{'x':>6}"
            elif grid[row][col] == '.' or grid[row][col] == 'I':
                # non-terminal
                action = policy[row][col]
                line += f"{action_map[action]:>6}"
            else:
                line += f"{'o':>6}"
        print(line)
# function that implements the policy iteration algorithm
def policy_iteration(grid, actions, reward, transition, gamma, K):
    rows, cols = len(grid), len(grid[0])
    U = np.zeros((rows, cols))
    # initialize pi with random, but legal actions
    def legal_actions(row, col):
        # get legal actions
        legal_a = []
        for action in actions:
            next_row, next_col = row, col

            if action == 'up':
                next_row -= 1
            elif action == 'down':
                next_row += 1
            elif action == 'left':
                next_col -= 1
            elif action == 'right':
                next_col += 1
            if 0 <= next_row < rows and 0 <= next_col < cols and grid[next_row][next_col] != 'X':
                legal_a.append(action)
        return legal_a
    def initialize():
        pi = np.empty((rows, cols), dtype=object)
        for row in range(rows):
            for col in range(cols):
                if isinstance(grid[row][col], (int, float)) or grid[row][col] == 'X':
                    pi[row][col] = 'up'
                else:
                    legal_a = legal_actions(row, col)
                    pi[row][col] = np.random.choice(legal_a)
        return pi
    pi = initialize()
    
    unchanged = False
    while not unchanged:
        U = policy_evaluation(grid, reward, gamma, pi, K, U, transition)
        unchanged = True
        for row in range(rows):
            for col in range(cols):
                max_value = float('-inf')
                best_action = None
                # compute max{∑[p(s'|s,a)U[s']]}
                for action in actions:
                    probs = transition((row, col), action)
                    value = sum(prob * U[next_r, next_c] for (next_r, next_c), prob in probs.items())
                    if value > max_value:
                        max_value = value
                        best_action = action
                # compute  ∑[p(s'|s,pi[s])U[s']]
                pi_action = pi[row][col]
                pi_probs = transition((row, col), pi_action)
                pi_value = sum(pi_prob * U[next_r, next_c] for (next_r, next_c), pi_prob in pi_probs.items())
                if max_value > pi_value:
                    pi[row][col] = best_action
                    unchanged = False
    return pi
# function that implements the policy evaluation
def policy_evaluation(grid, reward_func, gamma, pi, K, U, transition_model):
    U_k = np.copy(U)
    rows, cols = len(grid), len(grid[0])
    for k in range(1, K):
        U_k_prev = np.copy(U_k)
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 'X' and (grid[row][col] == '.' or grid[row][col] == 'I'):
                    action = pi[row][col]
                    probs = transition_model((row, col), action)
                    U_k[row, col] = reward_func(row, col) + gamma * sum(prob * U_k_prev[next_r, next_c] for (next_r, next_c), prob in probs.items())
                else:
                    U_k[row, col] = reward_func(row, col)
    return U_k