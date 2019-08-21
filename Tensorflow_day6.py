# ==============================================================================================================

# 신경망. DNN(Deep Neural Network)
# XOR problem

import tensorflow as tf

# x / y 데이터를 2차원으로 생성

x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]


y_data = [[0],
          [1],
          [1],
          [0]]


X = tf.placeholder(tf.float32,shape=[None, 2])
Y = tf.placeholder(tf.float32,shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) + b )

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*
                       tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# start training
for step in range(10001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# accuracy computation
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# hypothesis는 1의 값이 나오지 않기 떄문에(sigmoid 함수에서는 1에 한없이 가까워지는 것)
# 1의 값을 주기 위해 0.5보다 큰 값들을 1로 바꿔준다

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),
                                     dtype = tf.float32))

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)

# predict : test model

print(sess.run(predict, feed_dict = {X:x_data}))

"""
XOR 값도 계산을 못하는데, 더 깊은 문제를 머신러닝으로 어떻게 구현할 것인가?
"""

# ================================================================================================================

# 신경망을 쌓아서 만든 XOR 식

# x / y 데이터를 2차원으로 생성

x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]


y_data = [[0],
          [1],
          [1],
          [0]]


X = tf.placeholder(tf.float32,shape=[None, 2])
Y = tf.placeholder(tf.float32,shape=[None, 1])

# Layer 1
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)
"""
    열을 결정하면 다음 층의 행이 결정된다.
    (None, 2) * (2, ) = (None, )
"""

# Layer 2
W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2 )
"""
    (None, ) * ( , 1) = (None, 1)
    다층을 할 경우 이와 같이 결정되고 , 나머지는 미지수로 본인이 넣고 싶은 수를 넣는다
    Layer1:    (None, 2) * (2, 2) = (None, 2)
    Layer2:    (None, 2) * (2, 1) = (None, 1)
    이런 모형으로 결정된다.
"""


cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*
                       tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# start training
for step in range(10001):
    cost_val, W1_val, b1_val,\
              W2_val, b2_val,_ = \
        sess.run([cost, W1, b2, W2, b2, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 100 == 0:
        print(step, cost_val, W1_val, b1_val, W2_val, b2_val)


# accuracy computation
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# hypothesis는 1의 값이 나오지 않기 떄문에(sigmoid 함수에서는 1에 한없이 가까워지는 것)
# 1의 값을 주기 위해 0.5보다 큰 값들을 1로 바꿔준다

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),
                                     dtype = tf.float32))

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)

# predict : test model

print(sess.run(predict, feed_dict = {X:x_data}))