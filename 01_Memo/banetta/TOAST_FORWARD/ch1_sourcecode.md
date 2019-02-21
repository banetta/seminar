Jupyter Notebook
chap1_example1-1
Last Checkpoint: 2018.11.14
(autosaved)
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
1.1  Data 읽어오기: Constant 변수 생성
1.2  Model 정의
1.2.1  Constant 덧셈 계산 그래프 생성
1.3  Graph 정의
2  Run Graph
2.1  Session 생성
2.2  Constant 계산 Graph 실행
Example 1-1: 계산 그래프 생성과 실행 
import tensorflow as tf
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.logging.set_verbosity(tf.logging.ERROR)

print ('tensorflow ver.{}'.format(tf.__version__))
import tensorflow as tf
import numpy as np
​
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
tf.logging.set_verbosity(tf.logging.ERROR)
​
print ('tensorflow ver.{}'.format(tf.__version__))
tensorflow ver.1.10.0
1  Build Graph
1.1  Data 읽어오기: Constant 변수 생성
def get_data():
    node_a = tf.constant(3.0, dtype=tf.float32)
    node_b = tf.constant(4.0) # also tf.float32 implicitly
    return node_a, node_b
1.2  Model 정의
1.2.1  Constant 덧셈 계산 그래프 생성
tf.add() 이용

def build_model(X,Y):
#    node_add = X+Y
    node_add = tf.add(X,Y)
    
    return node_add
def build_model(X,Y):
    node_add = tf.add(X, Y)

    return node_add
1.3  Graph 정의
# 1. 데이터 읽기
node_a, node_b = get_data()
​
# 2. 모델 만들기
node_add = build_model(node_a,node_b)
2  Run Graph
2.1  Session 생성
TensorFlow는 계산 그래프 생성과 실행이 분리되어 있음
정의된 계산그래프는 session을 통해서만 연산 실행 및 결과를 얻을 수 있음
sess = tf.Session()
2.2  Constant 계산 Graph 실행
TensorFlow는 계산 그래프 생성과 실행이 분리되어 있음
정의된 계산그래프는 session을 통해서만 연산 실행 및 결과를 얻을 수 있음
add_result = sess.run(node_add)
​
print ('add result : {}'.format(add_result))
add result : 7.0
​
×
Drag and Drop
The image will be downloaded by Fatkun




Jupyter Notebook
chap1_example1-2
Last Checkpoint: 2018.11.14
(autosaved)
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
1.1  Model 정의
1.2  Graph 정의
2  Run Graph
2.1  Session 생성
2.2  계산 Graph 실행
2.3  placeholder에 array 형태의 입력 가능
Example 1-2: 계산 그래프 외부 입력 추가 
계산 그래프를 만들고, 계산그래프에 입력할 데이터를 동적으로 처리하기 위해서는 placeholder필요

tf.placeholder(dtype, shape=None, name=None)
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
print ('tensorflow ver.{}'.format(tf.__version__))
​
tf.logging.set_verbosity(tf.logging.ERROR)
tensorflow ver.1.10.0
1  Build Graph
1.1  Model 정의
입력받은 a, b에 대해 (𝑎+𝑏)×2 를 수행할 수 있는 계산그래프 생성
+, x OR tf.add(), tf.multiply() 이용

def build_model(node_a, node_b):
    # (a+b) 만들기
    node_add = tf.add(node_a,node_b)
    node_c = tf.constant(2.0)
    
    # (a+b)*2 만들기
    node_mul = tf.multiply(node_add,node_c)
​
    return node_mul
1.2  Graph 정의
# a, b 입력 부분생성
node_a = tf.placeholder(tf.float32)
node_b = tf.placeholder(tf.float32)
​
# 모델 만들기
model = build_model(node_a, node_b)
2  Run Graph
2.1  Session 생성
sess = tf.Session()
2.2  계산 Graph 실행
Session.run(실행 할 tensor list, feed_dict={placeholder:values})
feed_dict param. : placeholder 부분에 입력할 값을 {변수이름:값} 형태의 dict로 전달
2
result = sess.run([model], feed_dict={node_a:7, node_b:2})
​
print result
[18.0]
2.3  placeholder에 array 형태의 입력 가능
result = sess.run([model], feed_dict={node_a:[6, 2, 3], node_b:[4, 5, 6]})
print result
[array([20., 14., 18.], dtype=float32)]
​
×
Drag and Drop
The image will be downloaded by Fatku



Jupyter Notebook
chap1_example1-3
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
1.1  Model 정의: 𝑊𝑥+𝑏 model 만들기
1.2  Graph 정의
2  Run Graph
2.1  Session 생성 및 tf.Variable 변수들 초기화
2.2  계산 그래프 실행
Example 1-3: Training 할 수 있는 Tensor 만들기 
머신러닝의 training 과정에서 사용될 학습 파라미터들은 Variable로 생성

import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
tf.logging.set_verbosity(tf.logging.ERROR)
​
print ('tensorflow ver.{}'.format(tf.__version__))
tensorflow ver.1.10.0
1  Build Graph
1.1  Model 정의:  𝑊𝑥+𝑏  model 만들기
W * x + b
def build_model(x):
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    linear_model = W * x + b
    
    return linear_model
1.2  Graph 정의
# 외부입력 변수 설정
x = tf.placeholder(tf.float32)
​
# 모델 만들기
linear_model = build_model(x)
2  Run Graph
2.1  Session 생성 및 tf.Variable 변수들 초기화
sess = tf.Session()
​
init = tf.global_variables_initializer()
sess.run(init)
2.2  계산 그래프 실행
입력: 1, 2, 3, 4
결과:  0.3(1)−0.3,0.3(2)−0.3,0.3(3)−0.3,0.3(4)−0.3=0,0.3,0.6,0.9 
4
result = sess.run(linear_model, {x:[1, 2, 3, 4]})
​
print ("result: ", result)
('result: ', array([0.        , 0.3       , 0.6       , 0.90000004], dtype=float32))
​
×
Drag and Drop
The image will be downloaded by Fatkun





Jupyter Notebook
chap1_linear_regression_optim
Last Checkpoint: 2018.11.14
(autosaved)
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
 
Contents 
1  Build Graph
1.1  Data 읽어오기
1.2  Model 정의
1.2.1  - Placeholder: graph input 설정
1.3  Loss 정의
1.4  Opimizer 설정
1.5  Train Step 정의
1.6  Graph 정의
2  Run Graph
2.1  Session 생성 및 Variables 초기화
2.2  Graph 실행
Example: Linear Regression 
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
print ('tensorflow ver.{}'.format(tf.__version__))
tensorflow ver.1.10.0
1  Build Graph
1.1  Data 읽어오기
def get_data():
    data_x = np.reshape([[10, 1], [9, 1], [3, 1], [2, 1]], newshape=(4, 2))
    data_y = np.reshape([90, 80, 50, 30], newshape=(4, 1))
    return data_x, data_y
1.2  Model 정의
1.2.1  - Placeholder: graph input 설정
def build_model(X,W):
    hypothesis = tf.matmul(X, W) # Our Model
    
    return hypothesis
1.3  Loss 정의
def get_loss(Y, model):
    loss = tf.reduce_mean(tf.square(model-Y))
    
    return loss
1.4  Opimizer 설정
def get_optimizer():    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
​
    return optimizer
1.5  Train Step 정의
def get_train_step(loss, optimizer):
    train_step = optimizer.minimize(loss)
    
    return train_step
1.6  Graph 정의
## 데이터 읽어오기
data_x, data_y = get_data()
​
# 모델 만들기
X = tf.placeholder(tf.float32, shape=[4, 2])
Y = tf.placeholder(tf.float32, shape=[4, 1])
​
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
​
model = build_model(X,W)
​
# Loss 정의
loss = get_loss(Y, model)
​
# Optimizer 설정
optimizer = get_optimizer()
​
# Train Step 정의
train_step = get_train_step(loss, optimizer)
2  Run Graph
2.1  Session 생성 및 Variables 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
2.2  Graph 실행
for step in range(500):
    sess.run(train_step, feed_dict={X:data_x, Y:data_y})
    
    if (step) % 30 == 0:
        _loss, _W = sess.run([loss, W], feed_dict={X:data_x, Y:data_y})
        print '--step: {:3d}, W:({:.2f}, {:.2f}), Loss:{:.2f}'.format(step, _W[0][0], _W[1][0], _loss)
​
--step:   0, W:(6.62, 22.76), Loss:24.25
--step:  30, W:(6.61, 22.78), Loss:24.25
--step:  60, W:(6.61, 22.80), Loss:24.25
--step:  90, W:(6.61, 22.81), Loss:24.25
--step: 120, W:(6.61, 22.83), Loss:24.25
--step: 150, W:(6.61, 22.84), Loss:24.25
--step: 180, W:(6.61, 22.85), Loss:24.25
--step: 210, W:(6.61, 22.85), Loss:24.25
--step: 240, W:(6.61, 22.86), Loss:24.25
--step: 270, W:(6.60, 22.87), Loss:24.25
--step: 300, W:(6.60, 22.87), Loss:24.25
--step: 330, W:(6.60, 22.87), Loss:24.25
--step: 360, W:(6.60, 22.88), Loss:24.25
--step: 390, W:(6.60, 22.88), Loss:24.25
--step: 420, W:(6.60, 22.88), Loss:24.25
--step: 450, W:(6.60, 22.89), Loss:24.25
--step: 480, W:(6.60, 22.89), Loss:24.25
​
×
Drag and Drop
The image will be downloaded by Fatkun






Jupyter Notebook
chap1_logistic_classification
Last Checkpoint: 2018.11.14
(autosaved)
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
1.2  Model 정의
1.2.1  - Placeholder: graph input 설정
1.3  Loss 정의
1.4  Optimizer 설정
1.5  Train Step 정의
1.6  Graph 정의
2  Run Graph
2.1  Session 생성 및 Variables 초기화
2.2  Graph 실행
Example: Logistic Classfication 
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
print ('tensorflow ver.{}'.format(tf.__version__))
tensorflow ver.1.10.0
1  Build Graph
1.1  Data 읽어오기
def get_data():
    data_x = np.reshape([[10, 1], [9, 1], [3, 1], [2, 1]], newshape=(4, 2))
    data_y = np.reshape([1, 1, 0, 0], newshape=(4, 1))
    
    return data_x, data_y
1.2  Model 정의
1.2.1  - Placeholder: graph input 설정
def build_model(X,W):
    hypothesis = tf.sigmoid(tf.matmul(X, W))
    
    return hypothesis
1.3  Loss 정의
def get_loss(Y, model):
    loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model) + (1-Y)*(tf.log(1-model))))
    
    return loss
1.4  Optimizer 설정
def get_optimizer(lr_policy=None):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    return optimizer
1.5  Train Step 정의
def get_train_step(loss, optimizer):
    train_step = optimizer.minimize(loss)
​
    return train_step
1.6  Graph 정의
# 데이터 읽어오기
data_x, data_y = get_data()
​
# 모델 만들기
X = tf.placeholder(tf.float32, shape=[4, 2])
Y = tf.placeholder(tf.float32, shape=[4, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
​
# Model 정의
model = build_model(X,W)
​
# Loss 정의
loss = get_loss(Y, model)
​
# Optimizer 설정
optimizer = get_optimizer()
​
# Train Step 정의
train_step = get_train_step(loss, optimizer)
2  Run Graph
2.1  Session 생성 및 Variables 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
2.2  Graph 실행
for step in range(501):
    # Run graph
    sess.run(train_step, feed_dict={X:data_x, Y:data_y})
    if (step) % 30 == 0:
        _loss, _W = sess.run([loss, W], feed_dict={X:data_x, Y:data_y})
        print '--step: {:3d}, W:({:.2f}, {:.2f}), Loss:{:.2f}'.format(step, _W[0][0], _W[1][0], _loss)
--step:   0, W:(0.57, -2.68), Loss:0.64
--step:  30, W:(0.58, -2.78), Loss:0.62
--step:  60, W:(0.60, -2.87), Loss:0.59
--step:  90, W:(0.61, -2.95), Loss:0.56
--step: 120, W:(0.62, -3.03), Loss:0.54
--step: 150, W:(0.63, -3.11), Loss:0.52
--step: 180, W:(0.65, -3.19), Loss:0.50
--step: 210, W:(0.66, -3.26), Loss:0.48
--step: 240, W:(0.67, -3.34), Loss:0.46
--step: 270, W:(0.68, -3.40), Loss:0.45
--step: 300, W:(0.69, -3.47), Loss:0.43
--step: 330, W:(0.70, -3.53), Loss:0.42
--step: 360, W:(0.71, -3.60), Loss:0.40
--step: 390, W:(0.72, -3.66), Loss:0.39
--step: 420, W:(0.73, -3.72), Loss:0.38
--step: 450, W:(0.74, -3.77), Loss:0.37
--step: 480, W:(0.75, -3.83), Loss:0.36
data_test = np.reshape([[8, 1], [11, 1], [6, 1], [1, 1]], newshape=(4, 2))
print(sess.run(model,feed_dict={X:data_test}))
[[0.8975057 ]
 [0.98825616]
 [0.6595478 ]
 [0.04269456]]
​
×
Drag and Drop
The image will be downloaded by Fatkun







Jupyter Notebook
chap1_MLP
Last Checkpoint: 2018.11.14
(autosaved)
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
 
Contents 
1  Build Graph
1.1  Data 읽어오기
1.1.1  - MNIST Data 살펴보기
1.2  Model 정의
1.2.1  build model
1.3  Loss 정의
1.4  Optimizer 설정
1.5  Train Step 정의
1.6  Graph 정의
2  Run Graph
2.1  Session 생성 및 Variable 초기화
2.2  Graph 실행
2.3  Test
Example: Multi-Layer Perceptron (MLP) 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
​
print ('tensorflow ver.{}'.format(tf.__version__))
tf.logging.set_verbosity(tf.logging.ERROR)
tensorflow ver.1.10.0
1  Build Graph
1.1  Data 읽어오기
def get_data():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)
1.1.1  - MNIST Data 살펴보기
mnist = get_data()
​
print 'train image num : ', mnist.train.images.shape
print 'train label num : ', mnist.train.labels.shape
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
train image num :  (55000, 784)
train label num :  (55000, 10)
sample_idx = random.sample(range(mnist.train.images.shape[0]), 1)
​
sample_img = Image.fromarray(np.reshape(mnist.train.images[sample_idx]*255, (28, 28)).astype(np.int32))
sample_label = mnist.train.labels[sample_idx]
​
plt.imshow(sample_img)
print 'label : ', sample_label
label :  [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

1.2  Model 정의
1.2.1  build model
def build_model(x):
    W1 = tf.Variable(tf.random_normal([784, 128]), name='weight1')
    b1 = tf.Variable(tf.random_normal([128]), name='bias1')
    hidden_layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
​
    W2 = tf.Variable(tf.random_normal([128, 32]), name='weight2')
    b2 = tf.Variable(tf.random_normal([32]), name='bias2')
    hidden_layer2 = tf.nn.sigmoid(tf.matmul(hidden_layer1, W2) + b2)
​
    W3 = tf.Variable(tf.random_normal([32, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    logits = tf.matmul(hidden_layer2, W3) + b3
    prediction = tf.nn.softmax(logits)
    
    return {'logits': logits, 'prediction': prediction}
1.3  Loss 정의
def get_loss(y, model):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model['logits'], labels=y))
    return loss
1.4  Optimizer 설정
def get_optimizer():
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    return optimizer
1.5  Train Step 정의
def get_train_step(loss, optimizer):
    train_step = optimizer.minimize(loss)
​
    return train_step
1.6  Graph 정의
# 데이터 읽어오기
mnist = get_data()
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# `Placeholder`: graph input 설정
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
placeholder

* shape : [None, 784]
    * None
        * 정해지지 않음
        * 이미지를 여러장 씩 병렬처리
    * 784
        * 28x28이미지를 1-D로 표현
# Model 정의
model = build_model(x)
​
# Loss 정의
loss = get_loss(y, model)
​
# Optimizer 설정
optimizer = get_optimizer()
​
# Train Step 정의
train_step = get_train_step(loss, optimizer)
2  Run Graph
2.1  Session 생성 및 Variable 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
2.2  Graph 실행
batch : 동시에 처리하는 데이터 묶음
1 epoch : 전체 데이터셋을 한 번 training에 사용
1 step(iteration) : training 연산을 한 번 실행
MNIST train image 수 55,000
batch size : 100
1 epoch을 위해서는 55,000/100 = 550 step 수행 필요
train_images_size = mnist.train.images.shape[0]
# batch_size와 num_epoch를 변경하여 다른 훈련을 시킬 수 있다.
batch_size = 100
num_batch_per_epoch = int(math.ceil(train_images_size / batch_size))
num_epoch = 15
​
# Accuracy 정의
correct_prediction = tf.equal(tf.argmax(model['prediction'], 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
​
for idx in range(num_batch_per_epoch*num_epoch):
    # load data
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
​
    if idx % 100 == 0:
        _accuracy, _loss = sess.run([accuracy, loss], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print ('[{:04d} step] accuracy : {:.4f}, loss :{:.4f}'.format(idx, _accuracy, _loss))
    # train
    sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
[0000 step] accuracy : 0.9180, loss :0.2681
[0100 step] accuracy : 0.9176, loss :0.2689
[0200 step] accuracy : 0.9179, loss :0.2676
[0300 step] accuracy : 0.9180, loss :0.2685
[0400 step] accuracy : 0.9186, loss :0.2669
[0500 step] accuracy : 0.9183, loss :0.2671
[0600 step] accuracy : 0.9190, loss :0.2672
[0700 step] accuracy : 0.9185, loss :0.2675
[0800 step] accuracy : 0.9179, loss :0.2658
[0900 step] accuracy : 0.9184, loss :0.2670
[1000 step] accuracy : 0.9187, loss :0.2657
[1100 step] accuracy : 0.9194, loss :0.2659
[1200 step] accuracy : 0.9181, loss :0.2664
[1300 step] accuracy : 0.9195, loss :0.2646
[1400 step] accuracy : 0.9183, loss :0.2657
[1500 step] accuracy : 0.9187, loss :0.2663
[1600 step] accuracy : 0.9187, loss :0.2664
[1700 step] accuracy : 0.9190, loss :0.2639
[1800 step] accuracy : 0.9184, loss :0.2654
[1900 step] accuracy : 0.9196, loss :0.2635
[2000 step] accuracy : 0.9188, loss :0.2635
[2100 step] accuracy : 0.9205, loss :0.2630
[2200 step] accuracy : 0.9203, loss :0.2634
[2300 step] accuracy : 0.9198, loss :0.2631
[2400 step] accuracy : 0.9202, loss :0.2629
[2500 step] accuracy : 0.9194, loss :0.2630
[2600 step] accuracy : 0.9204, loss :0.2621
[2700 step] accuracy : 0.9191, loss :0.2627
[2800 step] accuracy : 0.9204, loss :0.2617
[2900 step] accuracy : 0.9196, loss :0.2622
[3000 step] accuracy : 0.9207, loss :0.2609
[3100 step] accuracy : 0.9210, loss :0.2617
[3200 step] accuracy : 0.9204, loss :0.2606
[3300 step] accuracy : 0.9197, loss :0.2621
[3400 step] accuracy : 0.9205, loss :0.2601
[3500 step] accuracy : 0.9201, loss :0.2593
[3600 step] accuracy : 0.9205, loss :0.2597
[3700 step] accuracy : 0.9210, loss :0.2592
[3800 step] accuracy : 0.9213, loss :0.2599
[3900 step] accuracy : 0.9206, loss :0.2594
[4000 step] accuracy : 0.9213, loss :0.2593
[4100 step] accuracy : 0.9209, loss :0.2608
[4200 step] accuracy : 0.9212, loss :0.2602
[4300 step] accuracy : 0.9213, loss :0.2582
[4400 step] accuracy : 0.9204, loss :0.2601
[4500 step] accuracy : 0.9211, loss :0.2590
[4600 step] accuracy : 0.9223, loss :0.2581
[4700 step] accuracy : 0.9210, loss :0.2579
[4800 step] accuracy : 0.9215, loss :0.2580
[4900 step] accuracy : 0.9215, loss :0.2579
[5000 step] accuracy : 0.9224, loss :0.2572
[5100 step] accuracy : 0.9222, loss :0.2575
[5200 step] accuracy : 0.9218, loss :0.2573
[5300 step] accuracy : 0.9216, loss :0.2575
[5400 step] accuracy : 0.9222, loss :0.2569
[5500 step] accuracy : 0.9214, loss :0.2568
[5600 step] accuracy : 0.9215, loss :0.2566
[5700 step] accuracy : 0.9213, loss :0.2563
[5800 step] accuracy : 0.9226, loss :0.2552
[5900 step] accuracy : 0.9230, loss :0.2553
[6000 step] accuracy : 0.9219, loss :0.2560
[6100 step] accuracy : 0.9222, loss :0.2552
[6200 step] accuracy : 0.9230, loss :0.2543
[6300 step] accuracy : 0.9230, loss :0.2547
[6400 step] accuracy : 0.9219, loss :0.2550
[6500 step] accuracy : 0.9229, loss :0.2548
[6600 step] accuracy : 0.9223, loss :0.2546
[6700 step] accuracy : 0.9220, loss :0.2545
[6800 step] accuracy : 0.9229, loss :0.2543
[6900 step] accuracy : 0.9231, loss :0.2532
[7000 step] accuracy : 0.9230, loss :0.2535
[7100 step] accuracy : 0.9237, loss :0.2535
[7200 step] accuracy : 0.9227, loss :0.2544
[7300 step] accuracy : 0.9231, loss :0.2529
[7400 step] accuracy : 0.9225, loss :0.2547
[7500 step] accuracy : 0.9231, loss :0.2527
[7600 step] accuracy : 0.9240, loss :0.2527
[7700 step] accuracy : 0.9240, loss :0.2527
[7800 step] accuracy : 0.9235, loss :0.2520
[7900 step] accuracy : 0.9227, loss :0.2518
[8000 step] accuracy : 0.9247, loss :0.2526
[8100 step] accuracy : 0.9236, loss :0.2520
[8200 step] accuracy : 0.9241, loss :0.2524
2.3  Test
test_idx = random.sample(range(mnist.test.images.shape[0]), 1)
​
test_img = Image.fromarray(np.reshape(mnist.test.images[test_idx]*255, (28, 28)).astype(np.int32))
test_label = mnist.test.labels[test_idx]
​
plt.imshow(test_img)
print 'gt label : ', np.argmax(test_label)
gt label :  1

_prediction = sess.run(model['prediction'], feed_dict={x:mnist.test.images[test_idx], y:mnist.test.labels[test_idx]})
​
prediction_label = np.argmax(_prediction[0])
prediction_score = _prediction[0][prediction_label]
print 'prediction label : {}, score: {:.4f}'.format(prediction_label, prediction_score)
prediction label : 1, score: 0.9990
​
×
Drag and Drop
The image will be downloaded by Fatkun


