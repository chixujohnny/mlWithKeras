# coding: utf-8

# 根据酒的特征预测这款酒的价格，回归问题
# github：https://github.com/sararob/keras-wine-model/blob/master/keras-wide-deep.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

# test your tf version
print('TensorFlow Version: ', tf.__version__)

# get data
URL = "https://storage.googleapis.com/sara-cloud-ml/wine_data.csv"
path = keras.utils.get_file(URL.split('/')[-1], URL)

# convert data to Pandas df
data = pd.read_csv(path)

# shuffle data
data = data.sample(frac=1)

# data preprocessing
data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1)

variety_threshold = 500  # variety出现低于这个量级的将被剔除
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index  # 需要remove的variety，如['Glera','Pinot Blanc']
data.replace(to_remove, np.nan, inplace=True)  # 将这些需要剔除的variety的数据置为空
data = data[pd.notnull(data['variety'])]  # 剔除所有variety中为空的样本

# split data to train and test
train_size = int(len(data) * 0.8)
print('train_size: ' + str(train_size) + '\n')
print('test_size:  ' + str(len(data) - train_size) + '\n')

# 拆分训练集测试集，因为上面已经做shuffle了，所以直接按顺序拆就行了
# train features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]
# train labels
labels_train = data['price'][:train_size]
# test features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]
# test labels
labels_test = data['price'][train_size:]

# # 对description文本特征做分词
# vocab_size = 12000  # 字典大小
# tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
# tokenize.fit_on_texts(description_train)  # only fit on train
#
# # wide feature1: sparse词袋，一共vocav_vector维度
# description_bow_train = tokenize.texts_to_matrix(description_train)
# description_bow_test = tokenize.texts_to_matrix(description_test)
#
# # wide feature2: 对varirty离散型特征做one-hot
# # 使用sklearn做LabelEncoder()，将n种标签值（一般是string类型）分配一个0到n-1之间的数字编码
# # 比如['apple','banana','banana','tomato','carrot']会编成[0,1,1,2,3]
# encoder = LabelEncoder()
# encoder.fit(variety_train)
# variety_train = encoder.transform(variety_train)
# variety_test = encoder.transform(variety_test)
# num_classes = np.max(variety_train) + 1  # variety一共有多少种
#
# # 将encoder好的variety数据转为one-hot
# # 比如将[0,1,1,2,3]变成
# #  [[ 1.  0.  0.  0.]
# #   [ 0.  1.  0.  0.]
# #   [ 0.  1.  0.  0.]
# #   [ 0.  0.  0.  1.]
# #   [ 0.  0.  1.  0.]]
# variety_train = keras.utils.to_categorical(variety_train, num_classes)
# variety_test = keras.utils.to_categorical(variety_test, num_classes)
#
# # 定义一下我们的wide模型结构
# bow_inputs = layers.Input(shape=(vocab_size, ))
# variety_inputs = layers.Input(shape=(num_classes, ))
# merged_layer = layers.concatenate( [bow_inputs, variety_inputs] )  # 将上面两个ont-hot型wide特征concat到一起
# merged_layer = layers.Dense(256, activation='relu')(merged_layer)
# predictions = layers.Dense(1)(merged_layer)
# wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)
# wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# # print(wide_model.summary())
#
# # deep model feature: word embedding of descreption
# train_embed = tokenize.texts_to_sequences(description_train)
# test_embed = tokenize.texts_to_sequences(description_test)
#
# max_seq_length = 170
# train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding='post')  # 在序列的结尾处补0，补到max_seq_length长度
# test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding='post')
#
# # 定义一下我们的deep模型结构
# deep_inputs = layers.Input(shape=(max_seq_length, ))
# embedding = layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=max_seq_length)(deep_inputs)
# embedding = layers.Flatten()(embedding)
# embed_out = layers.Dense(1)(embedding)
# deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
# deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# # print(deep_model.summary())
#
# # 将wide模型与deep模型concat起来
# merged_out = layers.concatenate( [wide_model.output, deep_model.output] )
# merged_out = layers.Dense(1)(merged_out)
# combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
# print(combined_model.summary())
#
# combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#
# # 开始训练
# combined_model.fit([description_bow_train, variety_train] + [train_embed], labels_train, epochs=10, batch_size=128)
# combined_model.evaluate([description_test, variety_test] + [test_embed], labels_test)