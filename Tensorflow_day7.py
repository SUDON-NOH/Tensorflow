# 참고파일 6일차
# MINIST(Modified National Institute of Standard Technology database)
# 손글씨체 이미지 : 28 * 28 = 784, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# train data : 55000개
# test data  : 10000개

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets('mnist_data/', one_hot = True)
# one-hot encoding - argmax 는 서로 반대의 관계이다.
# argmax는 가장 큰 값의 index 값을 출력
# one-hot encoding은 원본의 값들을 0 또는 1로 바꾸어준다.

print(mnist.train.images.shape) # (55000, 784) x 값
print(mnist.train.labels.shape) # (55000, 10)  y 값

print(mnist.test.images.shape) # (10000, 784) x 값
print(mnist.test.labels.shape) # (10000, 10)  y 값

plt.imshow(mnist.train.images[50000:50001].reshape(28,28),
           cmap='Greys', interpolation = 'nearest')
print(mnist.train.labels[50000:50001])
plt.show()

learning_rate = 0.01
training_epochs = 15
batch_size = 100 # 100개씩 꺼내서 학습시킨다.

tf.set_random_seed(777)

X = tf.placeholder(tf.float32, shape = [None, 784])
Y = tf.placeholder(tf.float32, shape = [None, 10]) # 이미 위에서 one hot 으로 인코딩 했기 때문

W = tf.Variable(tf.random_normal([784,10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')


#        (?, 784) * (784, 10) = (? 10)
logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                             labels = Y))
# cross_entropy
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Gradient 와 Adamoptimizer 둘 중 하나를 선택해서 한다. 둘 다 실행 후에 선택

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print("Start Learning!!")
# start training
for epoch in range(training_epochs) :  # 15
    avg_cost = 0
    # 550 = 55000/100
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch) : # 550회
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = { X:batch_xs , Y:batch_ys}
        c,_= sess.run([cost,optimizer],feed_dict = feed_dict )
        avg_cost += c / total_batch # 550번의 cost를 누적시켜서 550번으로 나누면 평균 cost가 나옴
    print('Epoch:','%04d'%(epoch + 1), 'cost:','{:.9f}'.format(avg_cost))
    # 55000개를 한번에 처리를 하면 GPU 메모리도 문제가 되지만 for 문을 한번 돌릴 때 W 값이 결정되기 때문에
    # Random으로 batch 값을 찾아서 for문을 돌려야한다. for문을 한번 돌릴 때 같은 값들만 들어온다면 W값에
    # 문제가 생기기 때문.
    # batch는 원본에서 size만큼을 추출해서 전체를 돌린다. 한번 for문을 수행한 후에는 돌린 것을 뺀 남은
    # 값들 중에서 batch를 수행한다.
print("Learning Finished!!")

# Test model and check accuracy
# accuracy computation
predict = tf.argmax(hypothesis,1) # 1 = 행으로 가서 찾아라 , 0 = 열으로 가서 찾아라
"""
hypothesis와 Y의 shape은 갖지만 갖고 있는 값들이 다르다.
hypothesis 는 0.2, 0.7, 0.4 등의 실수를 갖고 있고
Y는 0, 0, 0, 1, 0, 0 과 같이 0과 1로만 이루어져 있어 값이 다르기 때문에
두 값을 알맞게 변경해서 비교해야 한다. 그 방법이 argmax로 인덱스 값을 찾는 것이다.
"""
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                     dtype = tf.float32))
a = sess.run([accuracy],feed_dict={X:mnist.test.images ,Y:mnist.test.labels})
print('train images total number = ', mnist.train.num_examples) # 55000
print('test image total number = ', mnist.test.num_examples)    # 10000
print("\nAccuracy:",a)

# get one random test data and predict
r = random.randint(0,mnist.test.num_examples - 1) # 0 to 9999 random int number
print("random=",r, "Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1 )))

print("Prediction :", sess.run(tf.argmax(hypothesis,1),
                               feed_dict = { X: mnist.test.images[r:r+1]} ))

# matplotlib : imshow()
plt.imshow(mnist.test.images[r:r+1].reshape(28,28),
           cmap='Greys', interpolation='nearest')  # 2차원 보간법

plt.show()

# Epoch: 0012 cost: 0.289025393
# Epoch: 0013 cost: 0.284547325
# Epoch: 0014 cost: 0.284057295
# Epoch: 0015 cost: 0.280366603
# Learning Finished!!
# train images total number =  55000
# test image total number =  10000
#
# Accuracy: [0.9164]
# random= 4668 Label: [2]
# Prediction : [2]

