# ======================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy as cp
import sys

# ======================================================================================================================

x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for step in range(100):
    sess.run(train_op, feed_dict = {X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:' , sess.run(prediction, feed_dict={X: x_data}))
print('실제값:' , sess.run(target, feed_dict = {Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data,
                                                        Y: y_data}))

data = [[0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1]]

data_df = pd.DataFrame(data)

data_df.to_csv('data.csv', index = False)