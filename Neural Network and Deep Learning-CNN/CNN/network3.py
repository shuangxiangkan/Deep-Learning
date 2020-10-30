import numpy as np
import theano
import theano.tensor as T


from theano.tensor.nnet import sigmoid
from theano.tensor import shared_randomstreams


class FullyConnectedLayer(object):

    def __init__(self,n_in,n_out,activation_fn=sigmoid,p_dropout=0.0):
        self.n_in=n_in
        self.n_out=n_out
        self.activation_fn=activation_fn
        self.p_dropout=p_dropout
        # Initialize weights and biases
        self.w=theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0,scale=np.sqrt(1.0/n_out),size=(n_in,n_out)),
                dtype=theano.config.floatX),
            name='w',borrow=True)
        self.b=theano.shared(
            np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b',borrow=True
        )

    def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
        self.inpt=inpt.reshape((mini_batch_size,self.n_in))
        self.output=self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt,self.w)+self.b
        )
        self.y_out=T.argmax(self.output,axis=1)
        self.inpt_dropout=dropout_layer(
            inpt_dropout.reshape((mini_batch_size,self.n_in),self.p_dropout)
        )
        self.output_dropout=self.activation_fn(
            T.dot(self.inpt_dropout,self.w)+self.b
        )

    def accuracy(self,y):
        # Return the accuracy for the mini_batch
        return T.mean(T.eq(y,self.y_out))


def dropout_layer(layer,p_dropout):
    srng=shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999)
    )
    mask=srng.binomial(n=1,p=1-p_dropout,size=layer.shape)
    return layer*T.cast(mask,theano.config.floatX)


class Network(object):

    def __init__(self,layers,mini_batch_size):
        """
        Takes a list of  'layers', describing the network architecture, and
        a value for the 'mini_batch_size' to be used during training
        by stochastic gradient descent.
        """
        self.layers=layers
        self.mini_batch_size=mini_batch_size
        self.params=[param for layer in self.layers for param in layer.params]
        self.x=T.matrix("x")
        self.y=T.ivector("y")
        init_layer=self.layers[0]
        init_layer.set_input(self.x,self.x,self.mini_batch_size)
        for j in range(1,len(self.layers)):
            layer.set_input(
                prev_layer.output,prev_layer.output_dropout,self.mini_batch_size)
        self.output=self.layers[-1].output
        self.output_dropout=self.layers[-1].output_dropout
