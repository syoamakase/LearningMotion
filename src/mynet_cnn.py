from chainer import cuda, Variable, FunctionSet, optimizers, serializers, Chain
import chainer.links as L
import chainer.functions as F

class MyChain(Chain):
    data_output = []
    data_hidden = []
    data_first  = []
    def __init__(self,n_units,classnum,batchsize):
        super(MyChain, self).__init__(
                conv1=F.Convolution2D(1, 10, 2, stride=1, pad=1),
                conv2=F.Convolution2D(10, 30, 2, stride=1, pad=1),
                l1=F.Linear(2*2*30, n_units),
                l2=F.Linear(n_units, classnum),
            )
        self.batchsize = batchsize

    def __call__(self,x,y,state,train=True,target=True):
        h = Variable(x.reshape(len(x), 1, 4, 3), volatile=not train)
        t = Variable(y.flatten(), volatile=not train)

        h0 = F.max_pooling_2d(F.relu(self.conv1(h)), 2)
        h0 = F.max_pooling_2d(F.relu(self.conv2(h0)), 2)
        
        if target == False:
            data = h0.data
            self.data_first.append(data)
        
        h1= F.dropout(F.relu(self.l1(h0)),ratio=0.5,train=train)
        
        if target == False:
            data = h1.data
            self.data_hidden.append(data)
        
        y = self.l2(h1)

        if target ==False:
            data = y.data
            self.data_output.append(data)

        self.loss = F.softmax_cross_entropy(y,t)

        return state, self.loss
