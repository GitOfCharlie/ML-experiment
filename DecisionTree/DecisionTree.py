import numpy as np
import pandas as pd
from collections import Counter
import TreePlotter

def init_data(path):
    '''
    初始化数据
    :param path: 文件路径
    :return: DataFrame
    '''
    data_set = []
    cols = np.array(['age', 'prescript', 'astigmatic', 'tearRate', 'class'])
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            age, prescript, astigmatic, tearRate, category = line.split()
            data_set.append([age, prescript, astigmatic, tearRate, category])
    return pd.DataFrame(data_set, columns=cols)

def get_most_common_class(data_set: pd.DataFrame):
    '''
    计算最普遍的类
    :param data_set:
    :return: 类标记
    '''
    class_list = data_set['class']
    return Counter(class_list).most_common(1)[0][0]

def H(data_set: pd.DataFrame):
    '''
    计算经验熵H(D)
    :param data_set: 数据集D
    :return:
    '''
    class_list = data_set['class']
    class_values = set(class_list)
    H = 0.0
    for one_value in class_values:
        this_value_list = data_set[data_set['class'] == one_value]
        p_i = float(len(this_value_list)) / float(len(data_set))
        H -= p_i*np.log2(p_i)  # H(X) = -∑pi*log(pi)，取2为底的对数
    return H

def H_condition(data_set: pd.DataFrame, feature: str):
    '''
    计算经验条件熵H(D|A)，H(D|A) = ∑ pi*H(D|A=ai)
    :param data_set: 数据集D
    :param feature: 特征A
    :return:
    '''
    class_list = data_set['class']
    class_values = set(class_list)
    feature_values = set(data_set[feature])
    H_con = 0.0
    for one_feature_value in feature_values:
        this_value_list = data_set[data_set[feature] == one_feature_value]
        pi = float(len(this_value_list)) / float(len(data_set))
        H_con_this_value = H(this_value_list)
        H_con += pi * H_con_this_value
    return H_con



def get_feature_with_highest_Gain(data_set: pd.DataFrame):
    '''
    计算信息增益最大的特征及其增益量
    :param data_set: 数据集
    :return: 信息增益最大的特征，增益量
    '''
    features = data_set.columns.values[:-1]
    high_gain_feature = features[0]
    highest_gain = 0.0
    entropy = H(data_set)
    for feature in features:
        condition_entropy = H_condition(data_set, feature)
        gain = entropy - condition_entropy
        if gain > highest_gain:
            high_gain_feature = feature
            high_gain = gain
    return high_gain_feature, high_gain

def generate_decision_tree(data_set: pd.DataFrame, eps: float):
    '''
    生成决策树，迭代调用
    :param data_set: 当前数据集
    :param eps: ε信息增益阈值
    :return: 字典（子树）或单个值（叶子）
    '''
    # 分类结果列表
    class_list = np.array(data_set['class'])
    # 如果所有实例属于同一个类
    if sum(class_list == class_list[0]) == len(class_list):
        return class_list[0]
    # 如果标签集为空（只剩class列），则为单节点树，把实例数最大的类作为此节点标记类
    if len(data_set.columns.values) == 1:
        return get_most_common_class(data_set)
    # 获取信息增益最大的特征及其增益
    highest_gain_feature, highest_gain = get_feature_with_highest_Gain(data_set)
    # 增益小于ε，单一节点，返回实例数最大的类
    if highest_gain < eps:
        return get_most_common_class(data_set)
    # 构建树
    decision_tree_dict = {highest_gain_feature: {}}
    # 对每个最高增益特征的取值进行分割数据集，并进行递归调用生成树
    feature_values = set(data_set[highest_gain_feature])
    for one_value in feature_values:
        # 分割
        divided_data_set = data_set[data_set[highest_gain_feature] == one_value]
        #去除列
        divided_data_set = divided_data_set.drop(labels=highest_gain_feature, axis=1)
        # 生成子树
        decision_tree_dict[highest_gain_feature][one_value] = generate_decision_tree(divided_data_set, eps)
    return decision_tree_dict


if __name__ == '__main__':
    data_set = init_data('resources/lenses.txt')
    decision_tree = generate_decision_tree(data_set, eps=0.0001)
    print(decision_tree)
    TreePlotter.createPlot(decision_tree)
    # print(data_set[(data_set['tearRate'] == 'normal') & (data_set['astigmatic'] == 'yes') & (data_set['prescript'] == 'myope')])
    # print(data_set[data_set['tearRate'] == 'reduced'])

