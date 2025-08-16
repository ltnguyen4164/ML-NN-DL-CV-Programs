# Long Nguyen
# 1001705873

'''
Q-Learning function:
Q(s, a) = R(s) + γ∑[p(s'|s,a)maxQ(s',a')]
Input:
- (s,r,a): the previous state (s), the reward (r) obtained at state s
           the previous action (a) that led from s to s'
- (s', r'): the current state s', and the reward r' received at state s'
- gamma: the discount factor
- η: a function specifying a learning rate that decreases over time
Output:
- Q: the table of Q-values, storing utilities of (state, action) pairs
- N: a table, where N[s,a] counts all times that the agent was at state s
     AND chose action a as its next action
'''

from policy_iteration import policy_main
import numpy as np
import random

def Q_Learning_Update(grid, s, r, a, s_prime, r_prime, gamma, eta, Q, N):
    row, col = s_prime
    if isinstance(grid[row][col], float):
        Q[s_prime][None] = r_prime
    if s is not None:
        if (s, a) in N:
            # increment count of visit
            N[(s, a)] += 1
        else:
            # create new entry of visit
            N[(s, a)] = 1
        # call function η with input N[s,a]
        c = eta(N[(s, a)])
        max_q = max(Q[s_prime].values(), default=0)
        Q[s][a] = (1 - c) * Q[s].get(a, 0) + c * (r + gamma * max_q)
    return Q, N
def AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne):
    # function that parses the environment file and returns the grid, dimensions, and terminal rewards
    def parse_file(data_file):
        grid = []
        with open(data_file, 'r') as fp:
            for line in fp:
                grid.append([float(x.strip()) if x.strip().lstrip('-').replace('.', '', 1).isdigit() else x.strip() for x in line.strip().split(',')])
        return grid
    
    # define actions
    actions = ["up", "down", "left", "right"]
    
    # get environment
    grid = parse_file(environment_file)
    rows, cols = len(grid), len(grid[0])
    walls = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 'X'} 
    initials = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 'I'}
    valid = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 'X'}
    terminals = {(r, c) for r in range(rows) for c in range(cols) if isinstance(grid[r][c], float)}

    # η function
    def eta(N):
        return 20 / (19 + N)
    # f function
    def f(u, n):
        return 1 if n < Ne else u
    # function that returns current state s' and reward r'
    def SenseStateAndReward(state):
        row, col = state
        if grid[row][col] == 'I':
            # initial state == non terminal state
            return state, ntr
        elif grid[row][col] == '.':
            # non terminal state
            return state, ntr
        elif grid[row][col] == 'X':
            # blocked state
            return state, 0
        else:
            # terminal state
            return state, float(grid[row][col])
    # function that returns current state s' after action a
    def ExecuteAction(s, a):
        A = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),}
        perpendiculars = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"],}
        row, col = s
        probabilities = {}

        # check if state is terminal
        if isinstance(grid[row][col], float):
            return s
        
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

        # add perpendicular moves
        for perp_action in perpendiculars[a]:
            dx, dy = A[perp_action]
            next_row, next_col = row + dx, col + dy
            if is_valid(next_row, next_col):
                probabilities[(next_row, next_col)] = probabilities.get((next_row, next_col), 0) + 0.1
            else:
                probabilities[(row, col)] = probabilities.get((row, col), 0) + 0.1
        
        # select next state based on probabilities
        states, probs = zip(*probabilities.items())
        next_state = random.choices(states, weights=probs, k=1)[0]
        return next_state  
    
    # initialize Q, N to empty dictionaries (tables replacement)
    Q = {state: {action: 0.0 for action in actions} if state not in terminals else {None: 0.0} for state in valid}
    N = {}
    moves = 0
    
    # main loop: execute mission after mission until number of moves
    while moves < number_of_moves:
        s, r, a = None, None, None
        # choose randomly if multiple initial states exist
        s_prime = np.random.choice(list(initials)) if len(initials) > 1 else next(iter(initials))
        # execute one mission, from start to end
        while True:
            s_prime, r_prime = SenseStateAndReward(s_prime)
            Q, N = Q_Learning_Update(grid, s, r, a, s_prime, r_prime, gamma, eta, Q, N)
            # check if s' is terminal
            row, col = s_prime
            if isinstance(grid[row][col], float):
                # done with this mission
                break
            # add decaying random action frequency with epsilon
            epsilon = max(0.1, 1 - moves / number_of_moves)
            if random.random() < epsilon or all(Q[s_prime][action] == 0.0 for action in actions):
                a = random.choices(actions, weights=[N.get((s_prime, action), 1) for action in actions], k=1)[0]
            else:
                a = max(actions, key=lambda action: f(Q[s_prime][action], N.get((s_prime, action), 0)))
            next_state = ExecuteAction(s_prime, a)
            s, r = s_prime, r_prime
            s_prime = next_state
            moves += 1
    
    # print utilities
    print("utilities:")
    for row in range(rows):
        row_utilities = []
        for col in range(cols):
            if (row, cols) in walls:
                row_utilities.append(0.0)
            elif (row, col) in Q:
                if (row, col) in terminals:
                    utility = Q[(row, col)][None]
                else:
                    utility = max(Q[(row, col)].values())
                row_utilities.append(utility)
            else:
                row_utilities.append(0.0)
        print(" ".join(f"{utility:6.3f}" for utility in row_utilities))

    # call value_iteration program from assignment 7 for the policy functions only
    policy_main(environment_file, ntr, gamma, 20)