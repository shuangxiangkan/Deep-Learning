from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# model.summary()

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    # 目标目录
    'C:/Users/Kansx/.keras/datasets/cats_and_dogs_small/train',
    # 将所有图像的大小调整为150*150
    target_size=(150,150),
    batch_size=20,
    # 因为使用了binary_crossentropy损失，所以需要用二进制标签
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    # 目标目录
    'C:/Users/Kansx/.keras/datasets/cats_and_dogs_small/validation',
    # 将所有图像的大小调整为150*150
    target_size=(150,150),
    batch_size=20,
    # 因为使用了binary_crossentropy损失，所以需要用二进制标签
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('labels batch shape:',labels_batch.shape)
    break