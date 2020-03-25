import numpy as np
import pandas as pd
import time

# df = pd.DataFrame([[1, 3, 2], [4, 5, 6]]
#                   , index=[0, 1], columns=['a', 'b', 'c'])
# s = df.iloc[0, :]
# print(s.idxmax())
# N_STATES = 5
# s_ = 4
# reward = 1 if s_ == N_STATES else 0
# print('\raaa', end='')
# print('\rbbb', end='')
#
# idx = [(1, 2), (2, 3), (3, 4)]
# df = pd.DataFrame([[1], [2], [3]], index=idx)
# print(df)
#
#

ROWS = 4
COLS = 4
N_STATES = ROWS * COLS
ACTIONS = ['L', 'R', 'U', 'D']
EPS = 0.9
ALPHA = 0.1
LAMBDA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.3

def init_Q_table(n_rows, n_cols, actions):
    states = []
    for row in range(0, n_rows):
        for col in range(0, n_cols):
            states.append(str((row, col)))
    q_table = pd.DataFrame(
        data=np.zeros((len(states), len(actions))),
        columns=actions,
        index=states
    )
    return q_table

Q_table = init_Q_table(ROWS, COLS, ACTIONS)
print(Q_table)
curr_state = (1, 1)
actions_list = Q_table.loc[str(curr_state), :]
print(actions_list)

print('aaa\naaa\naaa\b\b\b\bbbb')
