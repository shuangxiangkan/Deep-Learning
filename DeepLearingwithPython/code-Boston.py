from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

# 因为需要将同一个模型多次实例化，所以用一个函数来构建模型
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

k=4
# "//"向下取整
num_val_samples=len(train_data)//k
num_epochs=100
all_scors=[]

for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
