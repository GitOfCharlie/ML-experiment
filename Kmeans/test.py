import numpy as np
import matplotlib.pyplot as plt
import random

# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a[:, 0])

# a = random.sample(range(0, 5), 5)
# print(a)

# a = np.array([1, 2])
# b = np.array([3, 4])
# print(np.sum(np.square(a - b)))

# plt.subplot(2, 2, 1)
# plt.plot([1, 2], [1, 2])
# plt.draw()
# plt.show()
#
# plt.subplot(2, 2, 2)
# plt.plot([1, 2], [1, 2])
#
# plt.show()


ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()                  # 开启一个画图的窗口
for i in range(100):       # 遍历0-99的值
	ax.append(i)           # 添加 i 到 x 轴的数据中
	ay.append(i**2)        # 添加 i 的平方到 y 轴的数据中
	plt.clf()              # 清除之前画的图
	plt.plot(ax,ay)        # 画出当前 ax 列表和 ay 列表中的值的图形
	plt.pause(0.1)         # 暂停一秒
	plt.ioff()             # 关闭画图的窗口
