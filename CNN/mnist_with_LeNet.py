# coding: utf-8

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

#  定义LeNet网络
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #  COV => ReLU => POOL
        #                卷积滤波器数量   每个滤波器大小5*5   保留边界处的卷积结果    2D卷积是我们的第一个阶段，所以必须定义input_shape
        model.add(Conv2D(filters=20,     kernel_size=5,     padding="same",        input_shape=input_shape                          ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #  COV => ReLU => POOL
        model.add(Conv2D(filters=50, kernel_size=5, padding="same"))  #  这次添加滤波器数量至50个，在更深的网络层中添加更多的filter数目是目前普遍的一个技术
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #  最后做flatten并接上全连接做softmax就可以做多分类了
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))  #  softmax
        model.add(Activation("softmax"))
        return model

#  定义参数
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1  # 日志参数，0：不输出日志，1：进度条形式输出，2：每个epoch输出一行
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

#  混合并划分训练集和测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#  把他们视为float类型，并归一化
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#  使用形状60K*(28,28,1)作为卷积网络的input
X_train = X_train[:, :, :, np.newaxis]  #  np.newaxis实质上就是创建一个新的轴，这里在最后一个位置添加一个轴也就是通道数(60000, 28, 28, 1)，教程参见https://blog.csdn.net/u013745804/article/details/79634196
X_test = X_test[:, :, :, np.newaxis]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#  将类向量转换成二值类别矩阵
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
#  初始化优化器和模型
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('Test score: ', score[0])
print('Test acc: ', score[1])

#  列出全部历史数据
print(history.history.keys())
#  汇总acc历史数据
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
#  汇总loss历史数据
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()