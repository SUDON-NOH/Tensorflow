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


#################
## 텐서보드 사용하기
#################

"""
딥러닝을 현업에 활용하게 되면 대부분의 경우 학습시간이 상당히 오래 걸린다.
따라서 모델을 효과적으로 실험하려면 학습 과정을 추적하는 일이 매우 중요하다.
하지만, 학습 과정을 추적하려면 번거로운 추가 작업을 많이 해야 한다.
텐서플로우는 이를 해결하기 위해 텐서보드라는 도구를 기본으로 제공한다.
"""


data = np.loadtxt('./data.csv', delimiter = ',',
                  unpack = True, dtype = 'float32', skiprows= 1)


x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

print(x_data)
# [[0. 0.]
#  [1. 0.]
#  [1. 1.]
#  [0. 0.]
#  [0. 0.]
#  [0. 1.]]


################
## 신경망 모델 구성
################


# 먼저 모델을 저장할 때 쓸 변수를 하나 만든다.
# 이 변수는 학습에 직접 사용되지는 않고, 학습 횟수를 카운트하는 변수이다.
# 때문에 trainable = False를 줬다.
global_step = tf.Variable(0, trainable = False, name = 'global_step')

# 편향 없이 가중치만을 사용한 모델로 만든다.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# 'with tf.name_scope' 로 묶은 블록은 텐서보드에서 한 계층 내부를 표현
# 'name = "W1"' 처럼 이름을 붙이면 텐서보드에서 해당 이름의 변수가 어디서 사용되는지 쉽게 알 수 있다.
# 이름은 변수뿐만 아니라 플레이스 홀더, 각각의 연산, 활서왛 함수 등 모든 텐서에 붙일 수 있다.
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name = 'W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name = 'W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name = 'W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels= Y, logits = model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step = global_step)


# 손실값을 추적하기 위해 수집할 값을 지정하는 코드를 작성
tf.summary.scalar('cost : ', cost)
"""
tf.summary 모듈의 scalar 함수는 값이 하나인 텐서를 수집할 때 사용한다.
물론 scalar 뿐만 아니라 histogram, image, audio 등 다양한 값을 수집하는 기본함수를 기본으로 제공.
"""

################
## 신경망 모델 학습
################

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# global_variables 는 모든 변수들을 가져오는 함수. 이 함수를 써서 앞서 정의한 변수들을 모두 가져와서,
# 이후 이 변수들을 파일에 저장하거나 이전에 학습한 결과를 불러와 담는 변수들로 사용합니다.



"""
아래의 코드는
'./model' 디렉터리에 기존에 학습해둔 모델이 있는지를 확인해서 모델이 있다면
saver.restore 함수를 사용해 학습된 값들을 불러오고, 아니면 변수를 새로 초기화 한다.

학습된 모델을 저장한 파일을 체크포인트파일(checkpoint file)이라고 한다.
"""
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# tf.summary.merge_all() 함수로 앞서 지정한 텐서들을 수집한 다음,
# tf.summary.FileWriter 함수를 이용해 그래프와 텐서들의 값을 저장할 디렉터리를 설정

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %d, ' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))


"""
sess.run을 이용해 앞서 merged 로 모아둔 텐서의 값들을 계산하여 수집한 뒤, writer.add_summary 함수를
이용해 해당 값들을 앞서 지정한 디렉터리에 저장한다. 적절한 시점에 값들을 수집하고 저장하면 되며, 나중에 확인할 수 있도록
global_step 값을 이용해 수집한 시점을 기록해둔다.
"""
summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
writer.add_summary(summary, global_step = sess.run(global_step))




saver.save(sess, './model/dnn.ckpt', global_step = global_step)
"""
두 번째 매개변수는 체크포인트 파일의 위치와 이름
global_step의 값은 저장할 파일의 이름에 추가로 붙게 되며, 텐서 변수 또는 숫자값을 넣어줄 수 있다.
이를 이용해 여러 상태의 체크포인트를 만들 수 있고, 가장 효과적인 체크포인트를 선별해서 사용할 수 있다.
"""

##########
## 결과 확인
##########

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))


# CH5_1 을 실행해 model 폴더가 이미 존재하면 이를 삭제하고 실행해야한다.
# 설정이 달라졌기 때문에 오류가 발생한다.

# ======================================================================================================================

"""
# 이를 실행하고 나면 현재 디렉터리에 logs 라는 디렉터리가 새로 생긴 것을 볼 수 있다.
# 윈도우 명령 프롬프트에서 다음 명령어를 입력한다.
# tensorboard --logdir=./logs
# 출력된 Starting TensorBoard b'41' on port 6006
# http://localhost:6006 의 형식으로 입력
"""

