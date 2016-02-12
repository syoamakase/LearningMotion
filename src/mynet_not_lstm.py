#-*-coding: utf-8 -*-

from chainer import cuda, Variable, FunctionSet, optimizers, serializers, Chain
import chainer.links as L
import chainer.functions as F


class MyChain(Chain):
    data_output = []
    data_hidden = []
    data_first  = []
    def __init__(self,n_units,classnum,batchsize):
        super(MyChain, self).__init__(
                l0=L.Linear(12, n_units),
                l1=L.Linear(n_units, n_units),
                l2=L.Linear(n_units,n_units),
                l3=L.Linear(n_units,classnum),
            )
        self.batchsize = batchsize
        #self.n_units  = n_units
        #self.classnum = classnum
                

    def __call__(self,x,y,state,train=True,target=True):
        if train:
            h = Variable(x.reshape(self.batchsize,12), volatile=not train)
        else:
            h = Variable(x, volatile=not train)
        
        t = Variable(y.flatten(), volatile=not train)
        
        h0 = F.dropout(F.relu(self.l0(h)),ratio=0.2,train=train)
        
        if target == False:
            data = h0.data
            self.data_first.append(data)
        
        h1= F.dropout(F.relu(self.l1(h0)),ratio=0.5,train=train)
        h2= F.dropout(F.relu(self.l1(h1)),ratio=0.4,train=train)
        if target == False:
            data = h1.data
            self.data_hidden.append(data)
        
        y = self.l3(h2)
        if target ==False:
            data = y.data
            self.data_output.append(data)
        self.loss = F.softmax_cross_entropy(y,t)
        return state,self.loss
