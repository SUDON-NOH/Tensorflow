# logistic_regression_day4.py

import tensorflow as tf

# x_data : [6,2]
x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

# y_data : [6,1]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W) + b )

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*
                       tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation : 정확성 측정
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),
                                     dtype = tf.float32))

# start training
for step in range(10001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)


# predict : test model
x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

print(sess.run(predict, feed_dict = {X:x_data}))