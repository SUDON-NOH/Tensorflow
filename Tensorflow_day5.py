# 신경망 구현

# 다층 : softmax
# 단층 : sigmoid

import tensorflow as tf
import numpy as np
"""
x_data = np.array(
    [[0,0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]])

# one-hot encoding :
# 데이터가 가질 수 있는 값들을 일려로 나열한 배열을 만들고,
# 그중 표현하려는 값을 뜻하는 인덱스의 원소만 1로 표기하고 나머지 원소는 모두 0으로 채우는 표기법


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
    tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y,
                                               logits = model))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict = {X:x_data, Y:y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict = {X:x_data}))
print('실측값:', sess.run(target, feed_dict={Y:y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' )
"""
# ==============================================================================================================

# softmax_zoo_multi_classification

xy = np.loadtxt('5일차/data-04-zoo.csv', delimiter=',', dtype = np.float32)
"""
numpy로 부를 때, 숫자만 부른다.
만약 파일에 주석처리가 되지 않은 부분이 있는 경우 skiprows로 해당 row를 빼고 불러야 한다.
"""

x_data = xy[:,:-1]   # 마지막행을 제외한 변수 값들만을 추출     # (101, 16)
y_data = xy[:,[-1]]  # 마지막행의 결과값만 추출               # (101, 1)
print(x_data.shape)
print(y_data.shape)

# Build Graph
# 변수
nb_classes = 7
X = tf.placeholder(tf.float32, shape = [None, 16])  # [?, 16] 주석을 입력할 땐 '?'를 None값을 대신해서 사용한다
Y = tf.placeholder(tf.int32, shape = [None, 1])   # Y의 값들이 One-hot encoding이 되어있지 않아서 바꿔야 한다.

# Y 값을 one_hot encoding으로 변환, Y값은 반드시 int형으로 입력
Y_one_hot = tf.one_hot(Y, nb_classes)                # [None, 1, 7]
print(Y_one_hot)                                     # shape=(?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # numpy에서 행을 자동으로 크기에 맞게 다시 조절한다 "-1"
print(Y_one_hot)                                     # [None, 1]

W = tf.Variable(tf.random_normal([16 , nb_classes], name = 'weight'))
b = tf.Variable(tf.random_normal([nb_classes], name = 'bias'))

# hypothesis
logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

# cost function
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                 labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습단계(start learning)
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict = {X : x_data, Y : y_data})
    if step % 100 == 0:
        print('step:', step,
              'cost_val:', cost_val,
              'W_val', W_val,
              'b_val', b_val)



# 검증단계(test, 정확도 측정)
# Accuracy Computation(정확도 계산)
predict = tf.argmax(hypothesis, 1) # 1:행단위 # 예측값 행에서 가장 큰 값, 확률값을 구한다.
# predict = tf.argmax(hypothesis,) # 0:열단위
"""
[0.01, 0.02, 0.03, 0.7, 0.1 ...]
  0     1      2    3    4
argmax는 가장 큰 값이 있는 인덱스 3을 추출해낸다.
"""
correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))  # predict와 y 값을 비교한다.
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32)) # 숫자로 바꾼 뒤 평균을 낸다.

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:x_data, Y:y_data})

print('Hypothesis:' , h,
      '\nPredict:',p,
      '\nAccuracy:', a)

# 예측단계(Predict)
pred = sess.run(predict, feed_dict = {X:x_data})
print('predict:', pred)

for p,y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} / Real Y: {}".format(p == int(y), p, int(y))) # 예측과 답이 같으면 True, 다르면 False
