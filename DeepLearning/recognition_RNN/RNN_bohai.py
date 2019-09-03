import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import globalVar as gl
import data_faction_for_rnn as df
import os
import pandas as pd
from sklearn.utils import shuffle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
train_data = shuffle(pd.read_csv(
    'out/trainRNNData/trainRNNData.csv', header=None))
test_data = shuffle(pd.read_csv(
    'out/testRNNData/testRNNData.csv', header=None))
tf.set_random_seed(1)

# 设置学习率，训练次数，批次大小，输入大小，时间步，神经元个数，输出等超级参数
lr = 0.001  # 学习率
train_inter = 2000000  # 训练次数
batch_size = 128  # 批次大小
lamda = 0.09  # 正则化率

n_inputs = 3  # 输入维度
n_steps = 16  # 时间步大小
n_neurons = 32  # 神经元个数
n_outputs = 2  # 输出维度
display_step = 20
# X, y placeholder
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# 对权重和偏置进行初始化定义
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_neurons])),
    'out': tf.Variable(tf.random_normal([n_neurons, n_outputs]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_neurons, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}


# 从数据集中获取下一批数据进行训练
def next_batch(train_flag, step, bat_size):
    # 获取一个文件中的所有数据
    if train_flag == 1:
        csv_data = train_data
    else:
        csv_data = test_data
    time = 0
    length = len(csv_data)
    if (step+1)*bat_size > length:
        if time == 0:
            time = step
        df = csv_data[(step % time)*bat_size:(step % time+1)*bat_size]
    else:
        df = csv_data[step*bat_size:(step+1)*bat_size]
    batch_ys = np.array(df[[48, 49]])
    batch_xs = np.array(df[list(range(48))])
    batch_xs = preprocessing.scale(batch_xs)
    batch_xs = np.reshape(batch_xs, [-1, n_steps, n_inputs])
    return batch_xs, batch_ys


# 定义lstm_model_swim模型
def lstm_model_swim(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_neurons])

    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_neurons, state_is_tuple=True)
    init_state = lstm_cell.zero_state(
        batch_size, dtype=tf.float32)  # 全部初始化为0 state
    outputs, _ = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=init_state, dtype=tf.float32)
    # 将outputs 变成列表
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # outputs = tf.reshape(output, [-1, n_neurons])
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
    tf.Graph()
    return pred


def trainLSTM(model_path):
    pred = lstm_model_swim(X, weights, biases)
    # 定义损失函数
    # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
    # 加入L2正则化
    tv = tf.trainable_variables()
    regularization_cost = lamda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred, labels=y)) + regularization_cost
    # 定义训练模型
    # 用GradientDescentOptimizer比Adam结果更可靠
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        while step * batch_size < train_inter:
            batch_xs, batch_ys = next_batch(1, step, batch_size)
            sess.run([train_op], feed_dict={X: batch_xs, y: batch_ys})
            if step % display_step == 0:
                los, acc = sess.run([loss, accuracy], feed_dict={
                                    X: batch_xs, y: batch_ys})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(los) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))
            step += 1
        saver.save(sess, model_path)
        print("Optimization Finished!")
        writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
        writer.close()

        # 测试
        step = 0
        accsum = 0
        totalsum = 0
        while step * batch_size < len(test_data):
            test_batch_xs, test_batch_ys = next_batch(0, step, batch_size)
            acc = sess.run(accuracy, feed_dict={
                           X: test_batch_xs, y: test_batch_ys})
            accsum += acc * batch_size
            totalsum += batch_size
            step += 1
            if step % display_step == 0:
                print("Testing..." + str(step))
        totalacc = accsum/totalsum
        print("Testing Accuracy= " + "{:.3f}".format(totalacc))
        return totalacc


# 预测
def prediction(model_path):
    saver = tf.train.Saver()
    pred = lstm_model_swim(X, weights, biases)
    # 定义损失函数
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        print("Model restored.")
        print('weights:')
        print(sess.run(weights['in']))
        print(sess.run(weights['out']))
        print('biases:')
        print(sess.run(biases['in']))
        print(sess.run(biases['out']))

        step = 0
        while step * batch_size < len(test_data):
            test_batch_xs, test_batch_ys = next_batch(0, step, batch_size)
            acc = sess.run(accuracy, feed_dict={
                           X: test_batch_xs, y: test_batch_ys})
            if step % display_step == 0:
                print("Step " + str(step) +
                      ", Testing Accuracy= " + "{:.3f}".format(acc))
            step += 1


if __name__ == "__main__":
    model_path = './SwimLSTMmodel.ckpt'
    trainLSTM(model_path)
    # prediction(model_path)
