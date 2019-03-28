# coding: utf-8

# 在main.py基础上增加了隐层，并使用ReLU激活

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1671) # 重复性设置

# 网络和训练参数
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # 多分类个数
OPTIMIZER = SGD() # 优化器
N_HIDDEN = 128 # 隐层单元个数
VALIDATION_SPLIT = 0.2 # 训练集中用作验证集的数据比例

# 数据：划分训练集、测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPE = 784 # 每个样本是28*28的二维矩阵，需要reshape()拉直
X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# 将类向量做one-hot
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# model
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE, )))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score: ", score[0])
print("Test accuracy: ", score[1])