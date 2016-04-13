from chainer import cuda, Variable, FunctionSet, optimizers, serializers, Chain
import chainer.functions  as F

class MyChain(Chain):
    data_output = []
    data_hidden = []
    data_first  = []
    def __init__(self,n_units,classnum,batchsize):
        super(MyChain, self).__init__(
                l0=F.Linear(12, n_units),
                l1_x=F.Linear(n_units, 4 * n_units),
                l1_h=F.Linear(n_units, 4 * n_units),
                l2_x=F.Linear(n_units, 4 * n_units),
                l2_h=F.Linear(n_units, 4 * n_units),
                l3=F.Linear(n_units,classnum),
            )
        self.batchsize = batchsize

    def __call__(self,x,y,state,train=True,target=True):
        if train:
            h = Variable(x.reshape(self.batchsize,12), volatile=not train)
        else:
            h = Variable(x, volatile=not train)
        
        t = Variable(y.flatten(), volatile=not train)
        
        h0 = F.relu(self.l0(h))
        
        if target == False:
            data = h0.data
            self.data_first.append(data)
        
        h1_in = self.l1_x(h0) + self.l1_h(state['h1'])
        h1_in = F.dropout(F.tanh(h1_in),train=train)
        c1, h1 = F.lstm(state['c1'], h1_in)
        h2_in = F.dropout(F.tanh(self.l2_x(h1)), train=train) + self.l2_h(state['h2'])
        c2, h2 = F.lstm(state['c2'], h2_in)

        if target == False:
            data = h1.data
            self.data_hidden.append(data)
        
        y = self.l3(h2)

        if target ==False:
            data = y.data
            self.data_output.append(data)
        state = {'c1': c1, 'h1': h1,'c2':c2,'h2':h2}
        self.loss = F.softmax_cross_entropy(y,t)

        return state,self.loss
