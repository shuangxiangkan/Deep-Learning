import numpy as np

k=4
num_validation_samples=len(data)//k

np.random.shuffe(data)

validation_scores=[]
for fold in rang(k):
    # 选择验证数据分区
    validation_data=data[num_validation_samples*fold:num_validation_samples*(fold+1)]
    training_data=data[:num_validation_samples*fold]+data[num_validation_samples*(fold+1):]

    # 创建一个全新的模型实例(未训练)
    model=get_model()
    model.train(training_data)
    validation_score=model.evaluate(validation_data)
    validation_scores.append(validation_score)

    # 最终验证分数：K折验证分数的平均值
    validation_score=np.average(validation_scores)

    # 在所有非测试数据上训练最终模型
    model=get_model()
    model.train(data)
    test_score=model.evaluate(test_data)