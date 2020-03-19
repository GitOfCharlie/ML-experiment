import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 加载数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
# mnist.train.images 为样本特征向量x，shape：(784, )
# mnist.train.labels 为结果向量y，shape：(10, )的一个ndarray

'''
建立placeholder，作为训练时的形参
'''
# 输入：28*28维向量
input_x = tf.placeholder(tf.float32, [None, 28*28])
# 输出：10维one-hot向量
output_y = tf.placeholder(tf.int32, [None, 10])

'''
定义CNN各层网络结构
'''
# 输入层
img_input_layer = tf.reshape(input_x, [-1, 28, 28, 1])
print("img_input_layer: ", img_input_layer)
# 卷积层1 使用双通道的conv2d便于处理图片，即卷积核窗口移动是二维的
conv1_layer = tf.layers.conv2d(
    inputs=img_input_layer,
    filters=32,  # 卷积核个数32，也就是卷积层的厚度
    kernel_size=[5, 5],  # 卷积核大小5×5
    strides=1,  # 每次窗口移动步数
    padding='same',  # 边界补'0'
    activation=tf.nn.relu  # 激活函数
)
print("conv1_layer: ", conv1_layer)
# 池化层1 使用最大值池化
pool1_layer = tf.layers.max_pooling2d(
    inputs=conv1_layer,
    pool_size=[2, 2],  # 每2×2的格子合并为1格
    strides=2  # 移动步数为2
)
print("pool1_layer: ", pool1_layer)
# 卷积层2
conv2_layer = tf.layers.conv2d(
    inputs=pool1_layer,
    filters=64,  # 卷积核个数64
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print("conv2_layer: ", conv2_layer)
# 池化层2
pool2_layer = tf.layers.max_pooling2d(
    inputs=conv2_layer,
    pool_size=[2, 2],
    strides=2
)
print("pool2_layer: ", pool2_layer)
# 平坦化层，经过两层卷积核池化层后形状变成7*7*64，将其扁平化成向量
flat_layer = tf.reshape(pool2_layer, [-1, 7*7*64])
print("flat_layer: ", flat_layer)
# 全连接层
full_connected_layer = tf.layers.dense(
    inputs=flat_layer,
    units=1024,  # 1024个单元
    activation=tf.nn.relu
)
print("full_connected_layer: ", full_connected_layer)
# 丢弃层，防止过拟合问题
dropout_layer = tf.layers.dropout(
    inputs=full_connected_layer,
    rate=0.3,  # 丢弃率
    training=True  # 在training模式中，返回dropout后的输出
)
print("dropout_layer: ", dropout_layer)
# 输出层 全连接
output_layer = tf.layers.dense(
    inputs=dropout_layer,
    units=10  # 10个神经单元
)
print("output_layer: ", output_layer)

'''
建立op操作
'''
# 计算误差op：使用交叉熵，用softmax计算百分比概率
loss_op = tf.losses.softmax_cross_entropy(
    onehot_labels=output_y,  # 原有标签y
    logits=output_layer  # 网络输出层的值
)
print("loss_op: ", loss_op)
# 训练op：梯度下降算法，学习率设为0.001
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_op)
print("train_op: ", train_op)
# 计算准确率op
accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(output_layer, axis=1)
)[1]
print("accuracy_op: ", accuracy_op)

'''
创建会话，初始化全局和局部变量，开始训练
'''
with tf.Session() as session:
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init)

    test_x = mnist.test.images  # 测试集的特征向量image
    test_y = mnist.test.labels  # 测试集的标签label
    # 训练迭代
    delta = 0.0002
    current_accuracy = 0.0
    similar_accuracy_times, max_similar_accuracy_times = 0, 3
    step_list, accuracy_list = [], []
    for i in range(100000):
        batch = mnist.train.next_batch(50)  # 获取下一个batch
        train_loss, train_op2 = session.run([loss_op, train_op], feed_dict={input_x: batch[0], output_y: batch[1]})
        if i % 100 == 0:
            test_accuracy = session.run(accuracy_op, feed_dict={input_x: test_x, output_y: test_y})
            step_list.append(i)
            accuracy_list.append(test_accuracy)
            print("Step = %d, Train loss = %.4f, Test accuracy = %.4f" % (i, train_loss, test_accuracy))
            # 检测收敛，如果连续max_similar_accuracy_times次测试集精确度的变化小于delta，则收敛
            if abs(test_accuracy - current_accuracy) < delta:
                similar_accuracy_times = similar_accuracy_times + 1
            else:
                similar_accuracy_times = 0
            if similar_accuracy_times == max_similar_accuracy_times:
                break
            current_accuracy = test_accuracy

    plt.plot(step_list, accuracy_list)
    # plt.title('迭代-精确度')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()

    # 测试：打印100个预测值和真实值对
    test_result = session.run(output_layer, {input_x: test_x[:100]})
    predict_y = np.argmax(test_result, axis=1)
    print('Predicted numbers', predict_y)  # 推测的数字
    print('Real numbers', np.argmax(test_y[:100], axis=1))
