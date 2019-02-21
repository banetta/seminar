# TOAST FORWARD 바로 실행해보는 딥러닝 기초 - NHN Machine Learning Lab

## Chapter 1 Deep Learning Basic -  김현기 사원
숫자 이미지 분류기

- 목차
  1. TensorFlow 소개
  2. 실습 환경 소개
  3. Basic Regression
  4. Basic Neural Network: MLP

- 왜 TensorFlow 인가?
  -   Google에서 만들고 유지보수!
  -   사용자가 많다!
      -   문제 발생시 구글링이 용이함
      -   새로운 모델 구현 및 지원이 빠름

Tensor + Flow
    Tensor: N 차원 배열

    3가지의 변수
    Placeholder : Graph의 입력 변수 매개 변수(argv 역할)
        x = tf.placeholder(tf.float32)...

        Session.run() 함수에 feed_dict

    Variable : Graph의 변수 Tensor 데이터 저장소 (in-memory 버퍼 역할)
        initialization 필수
        save & rode

    
[예제] TensorFlow Programming 1/2
    1. Building the computational graph
       1. node_a = tf.constant(3.0, )

    2. Running the computational graph
        sess = tf.session()

[예제] TensorFlow Programming 2/2
    #   build graph
    W = tf.Variable([.3], dtype = tf.float32)
    b = tf.Variable([-/3],dtype=tf.float32)

    x = tf.placeholder(tf.float32)

    linear_model - W * x + b

    #Run graph
    sess = tf.Session()
    init = tf.global_variable_initalizer()
    sess.run(init)
    .....

## 실습환경
    Jupyter notebook

    지정 쉘을 Run & 다음 쉘로 이동 : Shift + return


ML 이란?
    
    1. 주어진 데이터를 잘 표현할 수 있는 모델을 찾고(ex. 공부한 시간과 시험 점수 데이터, 공부한 시간에 따른 시험 점수 모델),
    2. 모델을 이용하여 새로운 데이터를 예측하는 방법!
        Q) 5시간 (x)을 

Linear Regrassion 예측 문제는 어떻게 학습시키나요?
    어떤 데이터에 대해 직선 그래프를 그린다.

Model 정하기 
    모델의 예측 값과 실제 값의 차이가 작을수록 좋은 모델
    Loss Functions : 예측 값과 실제 값의 차이
        최소로 하는것이 좋다!
        MSE(Mean Sen)

    e.g. Model 1,2,3 Loss 구하기

Model Training 과정
    목표 : 

Minimizing Loss Function (Error 줄이기)
    - Loss : Loss가 가장 가파르게 감소하는 방향 (기울기?)
    - a : learning rate 만큼 업데이트

[실습] Linear Regression
    공부한 시간 - 시험 점수 예측 모델 생성

Logistic Regression 분류 문제는 어떻게 학습 시키나요?
  -   공부한 시간에 따른 시험 합격/불합격 예측 문제
  -   합격할 확률을 알고 싶을 때

선형 함수로는 분류 문제 학습 불가하고, 계단 함수를 부드럽게 표현하여 Sigmoid 함수로 모델링을 진행한다.

Minimize Loss Function -> Learning
    Regression : 출력이 연속 값이어야 한다면
    MSE : 

Classification(분류) : 출력에 따라 분류를 해야한다면
    Cross Entropy Loss를 줄이는 방향으로 진행


[실습] Logistic Regression
    공부한 시간 - 시험 합격/불합격 예측 모델 생성

4. Basic Neural Network

    인간의 뇌와 닮은 뉴럴 네트워크
    Artificial neuron(Perceptron):여러개의 입력을 받고 하나의 값을 출력

    Activation Function 종류
    1. 선형 함수
    2. 계단 함수
    3. ReLu 함수
    4. tanh 함수 
    5. Sigmoid 함수

    Perceptron의 한계
    비선형 문제를 풀 수 없다. : 뉴런 한개(Linear Model)로 XOR 문제를 해결할 수 없다.
    2개의 뉴런을 붙이면? 가능!

    Multi-Layer Perceptron
    2개 이상의 층을 완전 연결(Fully-connected layer)
    -   입력 층(Input Layer
    -   중간 층(Hidden Layer)
    -   출력 층(Output Layer)

    Output Layer 설계
    예측 문제 vs 분류 문제
    출력을 그대로 사용 = 선형 함수
    해당 Class로 분류될 확률을 얻고 싶을 때 = Sigmoid 함수
    3개 이상의 class로 분류할 경우 =  Softmax 함수

    Softmax로 입력 X를 3개의 class로 구별하기
    -   각 연산은 Matrix연산으로 표현 가능

[실습] MLP Model 학습하기(MNIST)

일정 이상 같은 데이터로 반복학습을 하게 된다면 정답을 외워버리기 때문에 다른 새로운 데이터 셋이 들어오게 된다면 정확하지 못한 값을 줄 수 있다.


Machine Learning을 한다는 것은...

    데이터를 모아 가공하고, 데이터의 분포를 잘 표현할 수 있는 모델을 가정하고 모델을 평가할 수 있는 Loss 함수를 만들어서 프로그래밍을 하는것



## Convolutional Neural Network Basic - 석선희 선임 
패션 이미지 분류기

1.  Convolution Neural Network
2.  [실습] Fashion Item 분류기
3.  Well-known Architecture


이미지 인식을 위한 Neural Network
    MLP로 이미지 분류기를 만든다면?

    이미지에 적합한 Neural Network 만들어보자
        Convolution Layer: 이미지의 특징을 추출하기 위한 layer
        Pooling Layer: 추출된 이미지 특징 중 중요한 부분을 선택하기 위한 Layer

    이미지의 Convolution 연산 하는 방법
    Stp1. Filter와 이미지의 element-wise 곱셈의 합
    step2. Filter를 이용하여 

https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/

여러 개의 filter와 activation 함수를 통해 feature를 추출하는 layer


Quiz) AlexNet의 Convolution Layer는 몇개일까요? 5개

Quiz) AlexNet의 conv1 Parameter는 어떻게 구성되어 있을까?

Pooling Layer

[예제] CNN Model로 MNIST 학습하기

conv2 = tf.nn.conv2d(conv1, filters2)

[실습] Fashion MNIST 학습하기

간단한 Fashion 이미지 인식 모델 만들기

[실습] Training Param. Batch Size 결정

2. Well known architecture

이미지 인식 알고리즘 성능 측정을 위한 대회
