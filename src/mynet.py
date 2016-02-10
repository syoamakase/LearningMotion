#-*-coding: utf-8 -*-

from chainer import cuda, Variable, FunctionSet, optimizers, serializers, Chain
import chainer.functions  as F

batchsize = 40


class MyChain(Chain):
    data_output = []
    data_hidden = []
    data_first  = []
    def __init__(self,n_units,classnum):
        super(MyChain, self).__init__(
                l0=F.Linear(12, n_units),
                l1_x=F.Linear(n_units, 4 * n_units),
                l1_h=F.Linear(n_units, 4 * n_units),
                #l2_x=F.Linear(n_units, 4 * n_units),
                #l2_h=F.Linear(n_units, 4 * n_units),
                l2=F.Linear(n_units,classnum),
            )
        #self.n_units  = n_units
        #self.classnum = classnum
                

    def __call__(self,x,y,state,train=True,target=True):
        if train:
            h = Variable(x.reshape(batchsize,12), volatile=not train)
        else:
            h = Variable(x, volatile=not train)
        
        t = Variable(y.flatten(), volatile=not train)
        
        h0 = self.l0(h)
        
        if target == False:
            data = h0.data
            self.data_first.append(data)
        
        #h1_in = F.tanh(model.l1_x(h0)) + model.l1_h(state['h1'])
        h1_in = self.l1_x(h0) + self.l1_h(state['h1'])
        
        #使い方に関しては大丈夫(少なくとも表面上は)
        h1_in = F.dropout(F.tanh(h1_in),train=train)
        c1, h1 = F.lstm(state['c1'], h1_in)
        #h2_in = F.dropout(F.tanh(self.l2_x(h1)), train=train) + self.l2_h(state['h2'])
        #c2, h2 = F.lstm(state['c2'], h2_in)
        #h3 = F.dropout(F.tanh(self.l2_x(h2)), train=train,ratio=0.0)
        if target == False:
            data = h1.data
            self.data_hidden.append(data)
        
        y = self.l2(h1)
        if target ==False:
            data = y.data
            self.data_output.append(data)
        state = {'c1': c1, 'h1': h1}
        self.loss = F.softmax_cross_entropy(y,t)
        return state,self.loss
