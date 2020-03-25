import numpy as np
import pandas as pd
import time

np.random.seed(2)

'''
定义常量
'''
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
    '''
    初始化Q表
    :param n_rows:
    :param n_cols:
    :param actions:
    :return: index为元组的字符串形式表示坐标，column为上下左右四个动作
    '''
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

def get_next_action(curr_state: tuple, Q_table):
    '''
    获取下一个动作
    :param curr_state:
    :param Q_table:
    :return:
    '''
    actions_list = Q_table.loc[str(curr_state), :]
    # 1-eps概率随机选择动作（模拟退火），或初始情况下随机选择动作
    if np.random.uniform() > EPS or (actions_list == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        # 从最优动作中随机选一个动作
        action = np.random.choice(actions_list[actions_list == np.max(actions_list)].index)
    return action

def get_env_feedback(s: tuple, a: str):
    '''
    向左/上方向做一个小的惩罚（-0.1）来加快收敛，向右/下收益为0，右下角重点收益为1
    :param s:
    :param a:
    :return:
    '''
    # 向右
    if a == 'R':
        if s[1] == COLS-1:
            s_ = s
        else:
            s_ = (s[0], s[1] + 1)
        reward = 1 if s_[0] == ROWS - 1 and s_[1] == COLS-1 else 0
    # 向左
    elif a == 'L':
        if s[1] == 0:
            s_ = s
        else:
            s_ = (s[0], s[1] - 1)
        reward = -0.1
    elif a == 'U':
        if s[0] == 0:
            s_ = s
        else:
            s_ = (s[0] - 1, s[1])
        reward = -0.1
    else: # a == 'D'
        if s[0] == ROWS-1:
            s_ = s
        else:
            s_ = (s[0] + 1, s[1])
        reward = 1 if s_[0] == ROWS - 1 and s_[1] == COLS - 1 else 0
    return s_, reward

def update_env(s: tuple, episode, step_counter):
    # 更新环境状态

    #----
    #----
    #----
    #---T
    env_list = (['-']*COLS + ['\n'])*(ROWS-1) + ['-']*(COLS-1) + ['T']
    if s[0] == ROWS - 1 and s[1] == COLS - 1:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(1)
    else:
        env_list[(4+1)*s[0] + s[1]] = 'o'
        interaction = ''.join(env_list)
        print('{}'.format(interaction))
        print('------------------------------------------------')
        time.sleep(FRESH_TIME)

def Q_learning():
    Q_table = init_Q_table(ROWS, COLS, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = (0, 0)
        update_env(s, episode, step_counter)
        terminated = False
        while not terminated:
            a = get_next_action(s, Q_table)
            s_, r = get_env_feedback(s, a)
            Q_predict = Q_table.loc[str(s), a]
            # 如果走到终点
            if s_[0] == ROWS - 1 and s_[1] == COLS - 1:
                Q_actual = r
                terminated = True
            # 没走到终点
            else:
                Q_actual = r + LAMBDA * Q_table.loc[str(s_), :].max()

            # 更新Q表中Q(s, a)值
            Q_table.loc[str(s), a] += ALPHA * (Q_actual - Q_predict)
            s = s_
            step_counter += 1
            update_env(s, episode, step_counter)

    return Q_table

if __name__ == "__main__":
    Q_table = Q_learning()
    print('\r\nQ-table:\n')
    print(Q_table)