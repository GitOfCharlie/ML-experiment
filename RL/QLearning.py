import numpy as np
import pandas as pd
import time

np.random.seed(2)

'''
定义常量
'''
N_STATES = 8
ACTIONS = ['L', 'R']
EPS = 0.9
ALPHA = 0.1
LAMBDA = 0.9
MAX_EPISODES = 10
FRESH_TIME = 0.3

def init_Q_table(n_state, actions):
    q_table = pd.DataFrame(
        data=np.zeros((n_state, len(actions))),
        columns = actions
    )
    return q_table

def get_next_action(curr_state, Q_table):
    actions_list = Q_table.iloc[curr_state, : ]
    # 1-eps概率随机选择动作（模拟退火），或初始情况下随机选择动作
    if np.random.uniform() > EPS or (actions_list == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        # action = actions_list.idxmax()
        action = np.random.choice(actions_list[actions_list == np.max(actions_list)].index)
    return action

def get_env_feedback(s: int, a: str):
    # 向右
    if a == 'R':
        s_ = s + 1
        reward = 1 if s_ == N_STATES - 1 else 0
    # 向左
    else:
        if s == 0:
            s_ = s
        else:
            s_ = s - 1
        reward = -0.01
    return s_, reward

def update_env(s, episode, step_counter):
    # 更新环境状态
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if s == N_STATES-1:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(1)
        print('\r                                ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def Q_learning():
    Q_table = init_Q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        update_env(s, episode, step_counter)
        while s != N_STATES - 1:
            a = get_next_action(s, Q_table)
            s_, r = get_env_feedback(s, a)
            Q_predict = Q_table.loc[s, a]
            # 如果没走到终点
            if s_ != N_STATES - 1:
                Q_actual = r + LAMBDA * Q_table.iloc[s_, :].max()
            # 走到终点
            else:
                Q_actual = r
            # 更新Q表中Q(s, a)值
            Q_table.loc[s, a] += ALPHA * (Q_actual - Q_predict)
            s = s_
            step_counter += 1
            update_env(s, episode, step_counter)

    return Q_table

if __name__ == "__main__":
    Q_table = Q_learning()
    print('\r\nQ-table:\n')
    print(Q_table)