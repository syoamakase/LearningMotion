import numpy as np
import csv
import argparse
import sys
import time
import six
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers,serializers,Chain
from chainer import Link, Chain, ChainList
import chainer.functions  as F
import chainer.links as L
import matplotlib.pyplot as plt

import mynet
import mynet_not_lstm
import mynet_cnn

plt.style.use('ggplot')
mod = np

i_data = [10,11,12,26,27]
data_output = []
data_hidden = []
data_first  = []

batchsize = 120
n_units = 120
n_epoch = 70

bprop_len = 1
grad_clip = 1
classnum = len(i_data)

def load_csv(data_dir,data_file_name,num,test=False):
    data = []
    target = []
    with open(data_dir+"/"+data_file_name,"rU") as f:
        data_file = csv.reader(f,delimiter=",")
        for i, d in enumerate(data_file):
            data.append(np.array(d[1:]).astype(np.float32))
            target.append([num])
                        
    if test==False:
        train_data.extend(data)
        train_target.extend(target)
    else:
        test_data.extend(data)
        test_target.extend(target)
    return

def make_initial_state(batchsize=batchsize, train=True):
        return {name: Variable(mod.zeros((batchsize, n_units),
                                                 dtype=np.float32),
                                       volatile=not train)
                for name in ('c1', 'h1','c2','h2')}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='store',dest='data_dir',default='')
    parser.add_argument('--save', action='store',dest='save_filename',default='test')
    parser.add_argument('--gpu' ,'-g',default=01,type=int,help='GPU ID (negative value indicates CPU)')
    data_dir = parser.parse_args().data_dir
    save_filename = parser.parse_args().save_filename

    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []

    print "***load data ***"
    for x,i in enumerate(i_data):
        for j in xrange(1,8):
            for k in xrange(1,5):
                # files not exist
                if (i == 23 and j == 6 and k == 4) or (i==8 and j == 1 and k==4) or (i == 27 and j == 8 and k == 4):
                    pass
                else:
                    load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x)

    print "***load test ***"
    for x,i in enumerate(i_data):
        for j in xrange(8,9):
            for k in xrange(1,2):
                print eval("'a%d_s%d_t%d_.csv'%(i,j,k)")
                load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)

    state = make_initial_state()

    test_d   = np.array(test_data).astype(np.float32)
    test_t   = np.array(test_target).astype(np.int32)

    N      = len(train_data)
    N_test = len(test_data)
   
    model = mynet_cnn.MyChain(n_units,classnum,batchsize)
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    model.compute_accuracy = False

    optimizer = optimizers.RMSprop()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    def evaluate(x_data, t,target=True):
        state = make_initial_state(batchsize=len(t), train=False)
        state, loss = model(x_data, t, state, train=False,target=target)
        return loss.data.astype(np.float32)

    whole_len = len(train_data)
    jump = whole_len // batchsize
    print "jump = {}".format(whole_len)

    epoch = 0
    
    accum_loss = Variable(np.zeros(()).astype(np.float32))
    print('going to train {} iterations'.format(jump * n_epoch))

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Learning loop
    for i in xrange(jump * n_epoch):        
        x_batch = np.array([train_data[(jump * j+i) % whole_len]
                        for j in six.moves.range(batchsize)]).astype(np.float32)
        y_batch = np.array([train_target[(jump * j + i+1) % whole_len]
                        for j in six.moves.range(batchsize)]).astype(np.int32)

        state, loss = model(x_batch, y_batch, state)

        accum_loss.data =  accum_loss.data.astype(np.float32)
        accum_loss += loss

        if i % 1000 ==0:
            print('epoch = {} \n\ttrain loss: {}'.format(i,accum_loss.data))

        if (i + 1) % bprop_len == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = Variable(mod.zeros(()))
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        epoch += 1
            
        if (i+1) % (jump*n_epoch) == 0:
            evaluate_loss = evaluate(test_d, test_t,target=False)
        else:
            evaluate_loss = evaluate(test_d, test_t)
        if i % 1000 == 0:
            print('\ttest loss: {}'.format(evaluate_loss))
            
        sys.stdout.flush()

    serializers.save_hdf5(save_filename+'.model', model)
    serializers.save_hdf5(save_filename+'.state', optimizer)

