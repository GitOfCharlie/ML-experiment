import numpy as np
import matplotlib.pyplot as plt
import random


def init_random_points(num) -> np.ndarray:
    '''
    初始化num个二维坐标点
    :param num:点的个数
    :return:ndarray列表，包含点的坐标
    '''
    return np.random.uniform(low=0.0, high=10.0, size=(num, 2))

def init_k_centroids(points, k) -> np.ndarray:
    '''
    随机选择k个中心
    :param points: 点集
    :param k: 中心数量
    :return: 中心点列表
    '''
    length = len(points)
    centroids_idx = random.sample(range(length), k)
    centroids = []
    for i in centroids_idx:
        centroids.append(points[i])
    return np.array(centroids)

def cal_clusters(points, centroids) -> dict:
    '''
    计算每个点所属的簇
    :param points: 点集
    :param centroids: 中心点点集
    :return: (中心点，所属点列表)字典对象
    '''
    cluster_dict = dict()
    for point in points:
        distance = 0.0
        min_distance = float('inf')
        centroid_idx = -1
        for idx in range(len(centroids)):
            # 计算距离最近的中心点
            centroid = centroids[idx]
            distance = cal_distance(point, centroid)
            if(distance < min_distance):
                min_distance = distance
                centroid_idx = idx
        if centroid_idx not in cluster_dict.keys():
            cluster_dict[centroid_idx] = []
        cluster_dict[centroid_idx].append(point)

    return cluster_dict


def cal_distance(p1, p2):
    '''
    计算两点之间的欧式距离
    :param p1:
    :param p2:
    :return:
    '''
    return np.sqrt(np.sum(np.square(p1 - p2)))

def cal_loss(cluster_dict, centroids):
    '''
    计算损失函数值
    :param cluster_dict: 当前聚簇结果
    :param centroids: 中心点坐标集
    :return: 损失函数值
    '''
    loss = 0.0
    for centroid_idx in cluster_dict.keys():
        this_cluster = cluster_dict[centroid_idx]
        this_centroid = centroids[centroid_idx]
        for point in this_cluster:
            loss += cal_distance(point, this_centroid)
    return loss

def cal_centroids(cluster_dict) -> np.ndarray:
    '''
    重新计算中心点集
    :param cluster_dict: 当前聚簇结果
    :return:
    '''
    new_centroids = []
    for centroid_idx in cluster_dict.keys():
        this_cluster = cluster_dict[centroid_idx]
        new_centroid = np.mean(this_cluster, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


def plot_clusters(cluster_dict, centroids, iter_time):
    '''
    可视化
    :param cluster_dict: 当前聚类结果
    :param centroids: 中心点集
    :return: /
    '''
    color_mark = ['r', 'b', 'g', 'y', 'm', 'c']
    centroid_mark = ['r', 'b', 'g', 'y', 'm', 'c']

    plt.subplot(4, 3, iter_time + 1)
    plt.title('iteration ' + str(iter_time))
    for key in cluster_dict.keys():
        this_cluster = np.array(cluster_dict[key])
        plt.scatter(this_cluster[:, 0], this_cluster[:, 1], c=color_mark[key], s=20)

        centroid = centroids[key]
        plt.scatter(centroids[key][0], centroids[key][1], c=centroid_mark[key], s=200, marker=(5, 1), edgecolors='k')

    # plt.show()

if __name__ == '__main__':
    points = init_random_points(100)
    centroids = init_k_centroids(points, k=4)
    cluster_dict = cal_clusters(points, centroids);

    curr_loss = cal_loss(cluster_dict, centroids);
    prev_loss = float("inf")
    eps = 0.0001  # 迭代终止条件
    iter_time = 0;  # 迭代计数

    plt.figure(figsize=(15, 15))
    plot_clusters(cluster_dict, centroids, iter_time)

    while abs(curr_loss - prev_loss) > eps:
        # 迭代加一
        iter_time += 1
        # 更新上一次误差
        prev_loss = curr_loss
        # 更新中心点
        centroids = cal_centroids(cluster_dict)
        # 进行聚类
        cluster_dict = cal_clusters(points, centroids)
        # 计算误差
        curr_loss = cal_loss(cluster_dict, centroids)
        # 可视化过程
        plot_clusters(cluster_dict, centroids, iter_time)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

    print('各聚类中心点坐标:\n', centroids)
    print('各点聚类结果:\n', cluster_dict)
    for key in cluster_dict.keys():
        print('簇号', key, ':')
        for point in cluster_dict[key]:
            print(point)
