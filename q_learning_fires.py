import matplotlib.pyplot as plt
import numpy as np
import random
import time

alpha = 0.5
gamma = 0.999
epsilon = 0.1

np.random.seed(42)

# define utility function that generates the map
def draw_map(rows, cols):

    map = np.zeros((rows, cols))
    total_cells = rows * cols

    # choose cells on fire
    fire_prob_1 = np.random.choice(rows * cols, size=int(0.02 * total_cells), replace=False)
    fire_prob_1_rows, fire_prob_1_cols = np.unravel_index(fire_prob_1, (rows, cols))

    map[fire_prob_1_rows, fire_prob_1_cols] = 0.5

    return map

# define utility function that generates fires
def get_fires(map):

    map_rows = map.shape[0]
    map_cols = map.shape[1]

    # initialise fires matrix
    fires = np.zeros((map_rows, map_cols))

    for i in range(map_rows):
        for j in range(map_cols):
            fires[i,j] = np.random.choice([0, 1], size=1, p=[1 - map[i,j], map[i,j]])

    return fires

# define utility function that checks if a move is valid
def is_valid(row, col, map):

    if (0 <= row < map.shape[0]) & (0 <= col < map.shape[1]):
        return True

# define utility function that maps move to new location
# this serves as the state transition function
def get_new_loc(agent_loc, move, map):

    # get current row and current col from agent_loc
    current_row = agent_loc[0]
    current_col = agent_loc[1]

    if move == 0: # corresponds to up
        new_row = current_row - 1
        new_col = current_col
        if is_valid(new_row, new_col, map):
            new_loc = (new_row, new_col)
        else:
            new_loc = agent_loc

    if move == 1: # corresponds to down
        new_row = current_row + 1
        new_col = current_col
        if is_valid(new_row, new_col, map):
            new_loc = (new_row, new_col)
        else:
            new_loc = agent_loc

    if move == 2: # corresponds to left
        new_row = current_row
        new_col = current_col - 1
        if is_valid(new_row, new_col, map):
            new_loc = (new_row, new_col)
        else:
            new_loc = agent_loc

    if move == 3: # corresponds to left
        new_row = current_row
        new_col = current_col + 1
        if is_valid(new_row, new_col, map):
            new_loc = (new_row, new_col)
        else:
            new_loc = agent_loc

    return new_loc

# define utility function that creates a
# dictionary mapping from states to agent locations
def states_to_locs(states, cols):

    # initialise dictionary
    mapping = {}

    for state in range(states):
        row = state // cols
        col = state % cols
        mapping[state] = (row, col)


    return mapping

# define utility function that creates a
# dictionary mapping from agent locations to states
def locs_to_states(rows, cols):

    # initialise dictionary
    mapping = {}

    for i in range(rows):
        for j in range(cols):
            state_number = i * cols + j
            mapping[(i,j)] = state_number

    return mapping

# define utility function that
# initialises the Q-value matrix
def initialise_Q(state_space, action_space):

    Q = np.zeros((state_space, action_space))

    return Q

# define function that updates Q matrix
def update_Q(Q, map):

    Q_rows = Q.shape[0]
    Q_cols = Q.shape[1]

    map_rows = map.shape[0]
    map_cols = map.shape[1]

    # get dictionaries that allow us to
    # go between locations and states
    state_to_loc = states_to_locs(Q_rows*Q_cols, map_cols)
    loc_to_state = locs_to_states(map_rows, map_cols)

    # we iterate through every element in Q
    for i in range(Q_rows):
        # get current loc in state i
        current_loc = state_to_loc[i]
        for j in range(Q_cols):
            # get current Q value
            current_Q = Q[i,j]
            # get loc for if we take action j from state i
            new_loc = get_new_loc(current_loc, j, map)
            # get state corresponding to new loc
            new_state = loc_to_state[new_loc]
            # reward from taking action j from state i
            reward = np.random.choice([0, 1], size=1, p=[1 - map[new_loc], map[new_loc]])
            # to choose action from new_loc,
            # we use epsilon greedy
            epsilon_draw = random.random()
            if epsilon_draw < epsilon:
                # choose random move
                random_move = random.randint(0,3)
                # with this state and the random move,
                # look up Q-table for Q(s',a')
                Q_next = Q[new_state, random_move]
            else:
                # choose greedy move
                Q_next = max(Q[new_state, :])

            # update current Q value
            Q_update = current_Q + alpha*(reward + gamma*Q_next - current_Q)
            Q[i,j] = Q_update

    return Q

# define utility function to visualize
# the grid and the rewards
def visualize_map(map):

    # get map dims
    map_rows, map_cols = map.shape

    # Visualize the grid
    plt.imshow(map, cmap='viridis', interpolation='none', extent=[0, map_cols, 0, map_rows])

    plt.grid(True, color='black', linewidth=1, linestyle='-')

    # Show the plot
    plt.show()

# define utility function to visualize
# optimal policy
def visualize_q_values(Q_s_a, rows, cols):

    # Get the dimensions of the Q-values matrix
    num_states, num_actions = Q_s_a.shape

    # Find the action with the highest Q-value for each state
    max_actions = np.argmax(Q_s_a, axis=1)

    # Reshape the indices of max actions into a grid
    max_actions_grid = max_actions.reshape((rows, cols))

    # Create a custom colormap to map action indices to names
    cmap = plt.cm.get_cmap('viridis', num_actions)

    # Create a heatmap with the action names displayed
    plt.imshow(max_actions_grid, cmap=cmap, interpolation='none', extent=[0, cols, 0, rows], vmin=0,
               vmax=num_actions - 1)

    # Set colorbar ticks and labels based on unique actions
    cbar = plt.colorbar(ticks=np.arange(num_actions), label='Action with Highest Q-Value')

    # Display action names as tick labels
    cbar.set_ticklabels(['Up', 'Down', 'Left', 'Right'])

    plt.title('Action with Highest Q-Value for Each State')
    plt.show()


# define utility function that takes
# a Q-matrix and a state and reduces
# the Q-values at all predecessor
# (state,action) combinations for that
# location

def reduce_q(Q, map, loc):

    Q_rows = Q.shape[0]
    Q_cols = Q.shape[1]

    map_rows = map.shape[0]
    map_cols = map.shape[1]

    # get dictionaries to map b/w locs to states
    state_to_loc = states_to_locs(Q_rows * Q_cols, map_cols)
    loc_to_state = locs_to_states(map_rows, map_cols)

    # there are 4 possible predecessor locs
    # that can be accessed through the 4 moves

    up_pre_loc = get_new_loc(loc, 0, map)
    down_pre_loc = get_new_loc(loc, 1, map)
    left_pre_loc = get_new_loc(loc, 2, map)
    right_pre_loc = get_new_loc(loc, 3, map)

    # map to states
    up_pre_state = loc_to_state[up_pre_loc]
    down_pre_state = loc_to_state[down_pre_loc]
    left_pre_state = loc_to_state[left_pre_loc]
    right_pre_state = loc_to_state[right_pre_loc]

    # the action that needs updating for each
    # predecessor state is the opposite of the
    # one that we take to get to it from the
    # current state
    Q[up_pre_state, 1] = 0
    Q[down_pre_state, 0] = 0
    Q[left_pre_state, 3] = 0
    Q[right_pre_state, 2] = 0

    return Q


# define simulation function that takes
# as input the Q matrix, the map, and
# a starting location, and executes
# a policy for a specified number of
# steps, counting the reward along the way
def simulate(Q, map, current_loc, t, type):

    Q_rows = Q.shape[0]
    Q_cols = Q.shape[1]

    map_rows = map.shape[0]
    map_cols = map.shape[1]

    # get dictionaries that allow us to
    # go between locations and states
    state_to_loc = states_to_locs(Q_rows * Q_cols, map_cols)
    loc_to_state = locs_to_states(map_rows, map_cols)

    r = 0

    for i in range(t):

        # convert agent_loc to state
        current_state = loc_to_state[current_loc]

        if type == "Q":
            # look up Q table to get best move
            move = np.argmax(Q[current_state, :])
        else:
            random.seed(42)
            move = random.randint(0,3)

        # make move
        new_loc = get_new_loc(current_loc, move, map)

        # draw for fires in this timestep
        fires = get_fires(map)

        # get reward at new_loc
        r += fires[new_loc]*10

        # modify map to prevent future fires at
        # loc where fire was avoided
        #if fires[new_loc] == 1:
        #    map[new_loc] = 0

        # reduce the Q-value at (current_state, move)
        # to avoid returning here
        #Q[current_state, move] = 0

        # reduce the Q-value at all predecessor states
        # get state for new_loc
        #Q = reduce_q(Q, map, new_loc)

        # reset current_loc to new_loc for next move
        current_loc = new_loc

        print(current_loc)

    print(r)

def main():

     map = draw_map(100, 100)

     visualize_map(map)

     state_space = np.size(map)
     action_space = 4

     Q = initialise_Q(state_space, action_space)

     epsilon = 0.9

     for i in range(100):
         start_time = time.time()
         epsilon *= 0.9
         Q = update_Q(Q, map)
         end_time = time.time()
         runtime = end_time - start_time

     visualize_q_values(Q, 100, 100)

     simulate(Q, map, (49,49), 120, "Q")

     simulate(Q, map, (49, 49), 120, "random")




main()

