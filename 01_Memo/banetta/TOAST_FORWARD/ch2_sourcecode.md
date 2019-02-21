Jupyter Notebook
Example_01_MNIST
Last Checkpoint: 2018.11.14
(unsaved changes)
Current Kernel Logo
Python 2 
File
Edit
View
Insert
Cell
Kernel
Navigate
Widgets
LaTeX_envs
Help
 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
​
tf.logging.set_verbosity(tf.logging.ERROR)
1  Build Graph
1.1  Data 읽어오기¶
def get_data():
    return input_data.read_data_sets("datasets/mnist", one_hot=True)
1.2  Model 정의
1.2.1  Placeholder : graph 입력부분 정의
def get_inputs():
    # image data입력 부분
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
    # label data입력 부분
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    
    return x, y
1.2.2  Model : algorithm 을 graph 연산으로 정의
def get_model(images):
    
    x_image = tf.reshape(images, [-1, 28, 28, 1])
    
    # filter shape : w, h, in_channel, out_channel
    conv1_filters = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))
    conv1 = tf.nn.conv2d(x_image, conv1_filters, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    print 'conv1', conv1
​
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print 'pool1', pool1
​
    batch, h, w, d = [x.value for x in pool1.get_shape()]    
    flatten = tf.reshape(pool1, [-1, h*w*d])
    print 'flatten', flatten
    
    fc_weights = tf.Variable(tf.random_normal([h*w*d, 10], stddev=0.01))
    fc_bias = tf.Variable(tf.random_normal([10]))
    
    logits = tf.matmul(flatten, fc_weights) + fc_bias
    print 'logits', logits
    return logits
1.3  Loss 정의
def get_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
1.4  Optimizer 정의
def get_optimizer(lr): 
    return tf.train.GradientDescentOptimizer(learning_rate=lr)
1.5  Train Graph 정의
# 1. 데이터 읽기
mnist = get_data()
​
# 2. 모델과 모델입력부분 만들기
images, labels = get_inputs()
model_out = get_model(images)
​
# 3. Loss 만들기
loss = get_loss(model_out, labels)
​
# 4. Optimizer 만들기
optimizer = get_optimizer(lr=0.1)
​
train_op = optimizer.minimize(loss)
Extracting datasets/mnist/train-images-idx3-ubyte.gz
Extracting datasets/mnist/train-labels-idx1-ubyte.gz
Extracting datasets/mnist/t10k-images-idx3-ubyte.gz
Extracting datasets/mnist/t10k-labels-idx1-ubyte.gz
conv1 Tensor("Relu:0", shape=(?, 28, 28, 16), dtype=float32)
pool1 Tensor("MaxPool:0", shape=(?, 14, 14, 16), dtype=float32)
flatten Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
logits Tensor("add:0", shape=(?, 10), dtype=float32)
1.6  (Optional) Metric 정의
prediction = tf.nn.softmax(model_out)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
2  Run Graph
EPOCHS = 3
BATCH_SIZE = 100
NUM_BATCH_PER_EPOCH = int(mnist.train.images.shape[0]/float(BATCH_SIZE))
2.1  Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
​
for ep in range(EPOCHS):
    for st in range(NUM_BATCH_PER_EPOCH):
        batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)
        _, _acc, _loss = sess.run([train_op, accuracy, loss], feed_dict={images:batch_images, labels:batch_labels})
        
        if st % 100 == 0:
            print '{} Epoch, {} Step : acc({:.4f}), loss({:.4f})'.format(ep, st, _acc, _loss)
0 Epoch, 0 Step : acc(0.1000), loss(2.5194)
0 Epoch, 100 Step : acc(0.8300), loss(0.7859)
0 Epoch, 200 Step : acc(0.8700), loss(0.3375)
0 Epoch, 300 Step : acc(0.8700), loss(0.3649)
0 Epoch, 400 Step : acc(0.8900), loss(0.2724)
0 Epoch, 500 Step : acc(0.9200), loss(0.2755)
1 Epoch, 0 Step : acc(0.8800), loss(0.3374)
1 Epoch, 100 Step : acc(0.8700), loss(0.4353)
1 Epoch, 200 Step : acc(0.9400), loss(0.3201)
1 Epoch, 300 Step : acc(0.9400), loss(0.1481)
1 Epoch, 400 Step : acc(0.9200), loss(0.2372)
1 Epoch, 500 Step : acc(0.8900), loss(0.3489)
2 Epoch, 0 Step : acc(0.9100), loss(0.2863)
2 Epoch, 100 Step : acc(0.9500), loss(0.1794)
2 Epoch, 200 Step : acc(0.9600), loss(0.1845)
2 Epoch, 300 Step : acc(0.9400), loss(0.2698)
2 Epoch, 400 Step : acc(0.9400), loss(0.1597)
2 Epoch, 500 Step : acc(0.9800), loss(0.1383)
​
×
Drag and Drop
The image will be downloaded by Fatkun




Jupyter Notebook
Excercise_01_FashionMNIST
Last Checkpoint: 2018.11.14
(unsaved changes)
Current Kernel Logo
Python 2 
File
Edit
View
Insert
Cell
Kernel
Navigate
LaTeX_envs
Widgets
Help
 
Contents 
1  Build Graph
1.1  Data 읽어오기
1.1.1  Data 확인
1.2  Model 정의
1.2.1  Placeholder : graph 입력부분 정의
1.2.2  Model : algorithm 을 graph 연산으로 정의
1.3  Loss 정의
1.4  Optimizer 정의
1.5  Train Graph 정의
1.6  Metric 정의
2  Run Graph
2.1  Training
[실습] Fashion MNIST 학습하기 
목표 : 새로운 CNN모델을 만들어 최고 accuracy를 달성해주세요. 
제한시간 : 20분 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
​
import os
tf.logging.set_verbosity(tf.logging.ERROR)
1  Build Graph
1.1  Data 읽어오기
from tensorflow.examples.tutorials.mnist import input_data
​
def get_data():
    return input_data.read_data_sets("datasets/fashion", one_hot=True)
Fashion MNIST 
MNIST데이터 셋과 동일한 category수를 가지고 있으나 손글씨 숫자가 아닌 fashion item이미지로 구성
Fashion MNIST데이터의 category label은 다음과 같이 정의 되어 있음
Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
1.1.1  Data 확인
fashion_mnist = get_data()
test_images, test_labels = fashion_mnist.test.next_batch(1)
print len(test_images), len(test_labels), test_images[0].shape
plt.imshow(np.reshape(test_images[0], (28, 28)), cmap='gray')
print test_labels[0]
1.2  Model 정의
1.2.1  Placeholder : graph 입력부분 정의
placeholder 부분을 정의해주세요.
def get_inputs():
    x = ...
    y = ...
    
    return x, y
def get_inputs():
    x = tf.placeholder(tf.float32, [None, 28*28])
    y = tf.placeholder(tf.float32, [None, 10])

    return x, y
1.2.2  Model : algorithm 을 graph 연산으로 정의
모델을 자유롭게 구성해주세요.
def build_model(inputs):
    x_image = tf.reshape(inputs, [-1, 28, 28, 1])
        
    ...
​
    return logits
​
MNIST Example과 동일한 모델

def build_model(images):

  x_image = tf.reshape(images, [-1, 28, 28, 1])

  # filter shape : w, h, in_channel, out_channel
  conv1_filters = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))
  conv1 = tf.nn.conv2d(x_image, conv1_filters, strides=[1, 1, 1, 1], padding='SAME')
  conv1 = tf.nn.relu(conv1)
  print 'conv1', conv1

  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  print 'pool1', pool1

  batch, h, w, d = [x.value for x in pool1.get_shape()]    
  flatten = tf.reshape(pool1, [-1, h*w*d])
  print 'flatten', flatten

  fc_weights = tf.Variable(tf.random_normal([h*w*d, 10], stddev=0.01))
  fc_bias = tf.Variable(tf.random_normal([10]))

  logits = tf.matmul(flatten, fc_weights) + fc_bias
  print 'logits', logits

  return logits
1.3  Loss 정의
def get_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
1.4  Optimizer 정의
def get_optimizer(lr): 
    return tf.train.GradientDescentOptimizer(learning_rate=lr)
1.5  Train Graph 정의
fashion_mnist = get_data()
​
images, labels = get_inputs()
logits = build_model(images)
​
loss = get_loss(logits, labels)
optimizer = get_optimizer(lr=0.1)
​
train_op = optimizer.minimize(loss)
1.6  Metric 정의
prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
2  Run Graph
BATCH_SIZE = 100
EPOCHS = 3
​
NUM_BATCH_PER_EPOCH = int(fashion_mnist.train.images.shape[0]/float(BATCH_SIZE))
2.1  Training
training코드를 완성해주세요.
# session 만들기
sess = ...
# variables 초기화하기
sess.run(...)
​
for epoch in range(EPOCHS):
    for step in range(NUM_BATCH_PER_EPOCH):
        # batch_data 읽어오기
        batch_images, batch_labels = fashion_mnist.train.next_batch(BATCH_SIZE)
        # train_op실행하기
        sess.run(...)
        
        if step % 100 == 0:
            _accuracy, _loss = sess.run([accuracy, loss], feed_dict = {images:fashion_mnist.test.images, labels:fashion_mnist.test.labels})
            print '{} Epoch, {} Step : accuracy({:.4f}), loss({:.4f})'.format(epoch, step, _accuracy, _loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(EPOCHS):
    for step in range(NUM_BATCH_PER_EPOCH):
        # batch_data 읽어오기
        batch_images, batch_labels = fashion_mnist.train.next_batch(BATCH_SIZE)
        # train_op실행하기
        sess.run(train_op, feed_dict={images:batch_images, labels:batch_labels})

        if step % 50 == 0:
            _accuracy, _loss = sess.run([accuracy, loss], feed_dict = {images:fashion_mnist.test.images, labels:fashion_mnist.test.labels})
            print '{} Epoch, {} Step : accuracy({:.4f}), loss({:.4f})'.format(epoch, step, _accuracy, _loss)
×
Drag and Drop
The image will be downloaded by Fatkun