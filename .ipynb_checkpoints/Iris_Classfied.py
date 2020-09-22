'''
神经网络实现鸢尾花分类
1.准备数据
    1.1数据读入
    1.2数据乱序
    1.3生成训练集和测试集
    1.4配成“输入特征/标签”对，每次读入一小撮
2.搭建网络
    2.1初始化神经网络中所有可训练的参数
3.参数优化
    3.1嵌套循环迭代
    3.2with结构更新参数
    3.3显示当前loss
4.测试效果
    4.1计算当前参数前向传播后的准确率
    4.2显示当前acc
5.acc/loss可视化
'''


import tensorflow as tf
import numpy as np
from sklearn import datasets

# 1.1数据读入
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 1.2数据乱序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 1.3生成训练集和测试集
x_train = x_data[0:-30]
y_train = y_data[0:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 1.4配成“输入特征/标签”对，每次读入一小撮
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 2.1初始化神经网络中所有可训练的参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3],stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

print(x_train.shape)
print(b1)
print(np.array([-0.09194934, -0.12376948, -0.05381497]).shape)
