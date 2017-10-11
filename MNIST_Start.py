import tensorflow as tf

import inputData
mnist = inputData.read_data_sets("MNIST_data/",one_hot=True)

# 实现回归模型
x = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# 训练模型

y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 让tensorflow用梯度下降算法以0.01的学习速率，最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化创建的变量
init = tf.global_variables_initializer()

# 在一个Session里启动模型并初始化
sess = tf.Session()
sess.run(init)

# 开始训练模型
for i in range(1000):
    batch_xz, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xz, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
