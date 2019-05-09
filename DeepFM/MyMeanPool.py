# coding: utf-8

# 其功能是对2d或3d的tensor，指定一个axis进行求均值。例如[100,5,6]的矩阵，指定axis=1求均值，会变成[100,6]大小的矩阵。

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class MyMeanPool(Layer):

    def __init__(self, axis, **kwargs):  # **kwarg表示以字典形式传入预先未定义的任意数组命名参数
        self.supports_masking = True
        self.axis = axis
        super(MyMeanPool, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
