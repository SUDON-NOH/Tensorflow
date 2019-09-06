# Tensorflow
Tensorflow


# Overfitting
- Train accuracy와 Test accuracy를 비교한다.
- Train accuracy가 99% 나오는 상황에서 Test accuracy가 80% 정도 나오면 Overfitting으로 본다.
- 해결 방법
  1. 더 많은 Train data를 사용한다.
  1. feature의 개수를 줄인다.
  1. Regularization 한다.
  1. Dropout
  
### Regularization
- weight 을 구불구불하게 만들지 말고 펴진 형태로 만든다.
- l2reg = 0.001 * tf.reduce_sum(tf.square(W))

### Dropout : A Simple Way to Prevent Neural Networks from Overfitting
- 몇개의 neurons을 죽이자.  
keep_prob = tf.placaeholder("float")  
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))  
L1 = tf.nn.dropout(L1, keep_prob = keep_prob) # dropout 의 비율은 무작위로 결정된다. 보통 0.5 ~ 0.7로 사용  
....  
for epoch in range(training_epochs):  
&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;for i in range(total_batch):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;batch_xs, batch_ys = ....  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c, _ = sess.run([...], feed_dict = feed_dict)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;avg_cost ...  
   
 print('Accuracy:', sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  

TRAIN :  
 sess.run(optimizer, feed_dict = {X: batch_xs, Y: batch_ys, dropout_rate: 0.7})  
 학습할 때는 70프로만 사용하고
 
EVALUATION:  
 print "Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1})  
 실제 할 때는 100프로 모두 사용해야한다.
 
 
 ### Xavier for MNIST
 - W 값으 initializer 의 값을 tf.contrib.layers.xavier_initializer() 로 준다.
