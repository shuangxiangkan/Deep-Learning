import numpy as np
import random


class Network():

    def __init__(self,sizes):
        self.sizes=sizes
        self.length=len(sizes)
        self.biaes=[np.random.randn(b,1) for b in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def SGD(self,training_data,epoches,mini_batch_size,eta,test_data=None):
        training_data=list(training_data)
        training_data_length=len(training_data)

        if test_data:
            test_data=list(test_data)

        test_data_length=len(test_data)

        for i in range(epoches):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,training_data_length,mini_batch_size)]
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch,eta)

            if test_data:
                print("Epoch {} : {} / {}".format(i,self.evaluate(test_data),test_data_length))
            else:
                print("Epoch {} complete".format(i))



    def updata_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biaes]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_b,delta_w=self.backprop(x,y)
            nabla_b=[b+nb for b,nb in zip(nabla_b,delta_b)]
            nabla_w=[w+nw for w,nw in zip(nabla_w,delta_w)]

        self.biaes = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biaes, nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation=x
        activations=[activation]
        zs=[]
        for b,w in zip(self.biaes,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for i in range(2,self.length):
            z=zs[-i]
            delta=np.dot(self.weights[-i+1].transpose(),delta)*sigmoid_prime(z)
            nabla_b[-i]=delta
            nabla_w[-i]=np.dot(delta,activations[-i-1].transpose())

        return (nabla_b,nabla_w)

    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),y) for x,y in test_data]

        return sum(int(x==y) for (x,y) in test_results)


    def feedforward(self,a):
        for b,w in zip(self.biaes,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a


    def cost_derivative(self,activation, y):
        return  activation-y

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(activation,y):
    return y-activation
