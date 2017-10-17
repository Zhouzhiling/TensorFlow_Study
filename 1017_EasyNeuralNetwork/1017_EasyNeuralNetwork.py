import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置带噪声的线性数据
num_example = 50
# 生成一个完全线性的数据
X = np.array([np.linspace(-2, 4, num_example), np.linspace(-6, 6, num_example)])

# 数据展示
# plt.figure(figsize=(4, 4))
# plt.scatter(X[0], X[1])
# plt.show()

# 给数据增加噪声
X += np.random.randn(2, num_example)
# 数据展示
# plt.figure(figsize=(4, 4))
# plt.scatter(X[0], X[1])
# plt.show()

# 目标就是通过学习，找到一条拟合曲线，去还原最初的线性数据
# 把数据分离成 x 和 y
x, y = X
# 添加固定为1的bias
x_with_bias = np.array([(1., a) for a in x]).astype(np.float32)

# 用来记录每次迭代的loss
losses = []
# 迭代次数
training_steps = 50
# 学习率（步长），表示在梯度下降时每次迭代所前进的长度
learning_rate = 0.002

with tf.Session() as sess:
    # 设置所有张量，变量和操作
    # 输入层是x值和bias结点
    input = tf.constant(x_with_bias)
    # target是y的值，需要被调整成正确的尺寸
    target = tf.constant(np.transpose([y]).astype(np.float32))
    # weights是变量，每次循环会变，这里随机初始化(均值0，标准差0.1的高斯分布
    weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

    # 初始化所有变量
    tf.global_variables_initializer().run()

    # 设置循环中所要做的全部操作
    # 对于所有的x，根据现有的weights产生对应的y值，即 y = w2 * x + w1 * bias
    y_hat = tf.matmul(input, weights)
    # 计算误差，预计的y和真实的y之间的区别
    y_error = tf.subtract(y_hat, target)
    # 想要最小化L2损失，即误差的频发
    loss = tf.nn.l2_loss(y_error)
    # 上面的 loss 函数相当于
    # loss = 0.5 * tf.reduce_sum(tf.multiply(y_error, y_error))

    # 执行梯度下降
    # 更新 weights，比如weights += grads * learning_rate
    # 使用偏微分更新weights
    update_weights = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # 上面的梯度下降相当于
    # gradient = tf.reduce_sum(tf.transpose(tf.multiply(input, y——error)), 1, keep_dims=True)
    # update_weights = tf.assign_sub(weights, learning_rate * gradient)

    # 现在我们定义了所有的张量，初始化了所有操作（每次执行梯度优化）
    for _ in range(training_steps):
        # 重复跑，更新变量
        update_weights.run()
        # 记录每次迭代的loss
        losses.append(loss.eval())

    # 训练结束
    betas = weights.eval()
    y_hat = y_hat.eval()

# 展示训练趋势
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.3)
fig.set_size_inches(10, 4)
ax1.scatter(x, y, alpha=.7)
ax1.scatter(x, np.transpose(y_hat)[0], c="g", alpha=.6)
line_x_range = (-4, 6)
ax1.plot(line_x_range, [betas[0] + a * betas[1] for a in line_x_range], "g", alpha=.6)

ax2.plot(range(0, training_steps), losses)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Training steps")
plt.show()
