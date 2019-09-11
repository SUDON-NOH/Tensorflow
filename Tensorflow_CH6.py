# ======================================================================================================================

###############################
#### 이미지 인식 CNN - 합성곱 신경망
###############################


"""
# 컨볼루션 계층 Convolution layer (합성곱 계층)
   - 2차원의 평면 행렬에서 지정한 영역의 값들을 하나의 값으로 압축하는 것
   - 하나의 값으로 압축할 때 컨볼루션 계층은 가중치와 편향을 적용
   - 윈도우 크기만큼의 가중치와 1개의 편향을 적용
   - 예를 들어 윈도우 크기가 3 x 3 이라면, 3 x 3 개의 가중치와 1개의 편향이 필요

# 풀링 계층 pooling layer
   - 단순히 값들 중 하나를 선택해서 가져오는 방식을 취함

# 윈도우
   - 지정한 크기의 영역

# 스트라이드 Stride
   - 몇 칸씩 움직일지 정하는 값
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)

# 1 ====================================================================================================================

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
"""
 - X의 None 은 입력 데이터의 개수
 - X의 1은 특징의 개수로, MNIST 데이터는 회색조 이미자ㅣ라 채널에 색상이 한 개 뿐이므로 1을 사용
"""

# 2 ====================================================================================================================
# 출력값은 10개의 분류

Y = tf.placeholder(tf.float32, [None, 10])


keep_prob = tf.placeholder(tf.float32) # Dropout을 위한 placeholder

# 3 ====================================================================================================================

# CNN 계층 구성
# 3 x 3 크기의 커널을 가진 컨볼루션 계층을 만든다.

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
L1 = tf.nn.conv2d(X, W1, strides = [ 1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)

"""
입력층 X와 첫 번째 계층의 가중치 W1을 가지고, 오른쪽과 아래쪽으로 
한칸씩 움직이는 32개의 커널을 가진 컨볼루션 계층을 만들겠다는 코드이다.

padding = 'SAME'은 커널 슬라이딩 시 이미지의 가장 외각에서 한 칸 밖으로 움직이는 옵션
이미지의 테두리까지도 좀 더 정확하게 평가가 가능하다.
"""

# 3 ====================================================================================================================

# 풀링 계층 생성

L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

"""
앞서 만든 컨볼루션 계층을 입력층으로 사용하고, 커널 크기를 2 x 2 로 하는 풀링 계층을 만든다.
strides = [ 1, 2, 2, 1 ] 값은 슬라이딩 시 두 칸씩 움직이겠다는 옵션
"""

# 4 ====================================================================================================================

# 두 번째 계층

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

"""
W2의 32는 앞서 구성한 첫 번째 컨볼루션 계층의 커널 개수이다.
출력층의 개수이며 또한 첫 번째 컨볼루션 계층이 찾아낸 이미지의 특징 개수라고 할 수 있다.
"""

# 5 ====================================================================================================================

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev = 0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

"""
먼저 10개의 분류는 1차원이므로 차원을 줄이는 단계를 거쳐야한다.
직전 풀링 계층 크기가 7 * 7 * 64 이므로, 먼저 tf.reshape 함수를 이용해
7 * 7 * 64 크기의 1차원 계층을 만들고, 이 배열 전체를 최종 출력값의 중간 단계인
256개의 뉴런으로 연결하는 신경망을 만들어준다.

   # 이처럼 인접한 계층의 모든 뉴런과 상호연결된 계층을 "완전 연결 계층"이라고 한다.
   
과적합 방지를 위한 dropout 사용
"""

# 6 ====================================================================================================================

# 직전의 은닉층인 L3의 출력값 256개를 받아 최종 출력값인 0 ~ 9 레이블을 갖는 10개의 출력값을 만든다.
W4 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01))
model = tf.matmul(L3, W4)

# 7 ====================================================================================================================

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# 8 ====================================================================================================================

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        # 모델에 입력값을 전달하기 위해 MNIST 데이터를 28 x 28 형태로 재구성하는 부분만 다르다.
        # batch_xs.reshape(-1, 28, 28, 1)
        # mnist.test.images.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict= {X: batch_xs,
                                           Y: batch_ys,
                                           keep_prob: 0.7})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict= {X: mnist.test.images.reshape(-1, 28, 28, 1),
                                   Y: mnist.test.labels,
                                   keep_prob: 1}))


