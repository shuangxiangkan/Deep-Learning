import mnist_loader

training_data,validation_data,test_data=mnist_loader.load_data_wrapper()

import network2

net=network2.Network([784,10])
net.SGD(training_data[:1000],30,10,10.0,lmbda=1000.0,
        evaluation_data=validation_data[:100],monitor_evaluation_accuracy=True)