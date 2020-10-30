import numpy as np

data=[1,2,3,4]
num_validation_samples=10000

# 通常需要打乱数据
np.random.shuffle(data)

# 定义验证集
validation_data=data[:num_validation_samples]
data=data[num_validation_samples:]

# 定义训练集
training_data=data[:]

# 在训练集上训练模型，并在验证数据上评估模型
model=get_model()
model.train(training_data)
validation_score=model.evaluate(validation_data)

# 现在你可以调节模型、重新训练、评估，然后再次调节


# 一旦调节好超参数，通常就在所有非测试数据上从头开始训练最终模型
model=get_model()
model.train(np.concatenate([training_data,validation_data]))
test_socre=model.evaluate(test_data)