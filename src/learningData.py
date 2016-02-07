#!/usr/bin/env python
#-*-coding:utf-8 -*- 
##データをロードする

import numpy as np
import csv
import argparse
import sys
import time
import six
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers,serializers,Chain
import chainer.functions  as F
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mod = np

i_data = [10,12,26,27]
data_output = []
data_hidden = []
data_first  = []


#バッチサイズ(60:微妙 20:微妙)
batchsize = 40
#中間層(隠れ層)の個数
n_units = 250
#n_units = 250
#batchsize = 40
#学習回数
n_epoch = 40
#n_epoch = 20

#BPTTの長さ
bprop_len = 1
#絶対値クリッピングの値
grad_clip = 1    # gradient norm threshold to clip
#分類クラス
classnum = len(i_data)

# csvファイルを読み込む関数
def load_csv(data_dir,data_file_name,num,test=False):
    data = []
    target = []
    with open(data_dir+data_file_name,"rU") as f:
        data_file = csv.reader(f,delimiter=",")
        data_length = 0
        for i, d in enumerate(data_file):
            if i == 0:
                prev_data = np.array(d).astype(np.float32)
            else:
                ##速度を出すための処理
                #data.append(np.array(d[1:]).astype(np.float32)-prev_data[1:])
                #prev_data= np.array(d).astype(np.float32)
                ##そのままのデータ(座標)を使う処理
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
                for name in ('c1', 'h1')}


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
                l0=F.Linear(12, n_units),
                l1_x=F.Linear(n_units, 4 * n_units),
                l1_h=F.Linear(n_units, 4 * n_units),
                #l2_x=F.Linear(n_units, 4 * n_units),
                #l2_h=F.Linear(n_units, 4 * n_units),
                l2=F.Linear(n_units,classnum),
            )

    def __call__(self,x,y,state,train=True,target=True):
        if train:
            h = Variable(x.reshape(batchsize,12), volatile=not train)
        else:
            h = Variable(x, volatile=not train)
        
        t = Variable(y.flatten(), volatile=not train)
        
        h0 = model.l0(h)
        
        if target == False:
            data = h0.data
            data_first.append(data)
        
        #h1_in = F.tanh(model.l1_x(h0)) + model.l1_h(state['h1'])
        h1_in = model.l1_x(h0) + model.l1_h(state['h1'])
        
        #使い方に関しては大丈夫(少なくとも表面上は)
        h1_in = F.dropout(F.tanh(h1_in),train=train)
        c1, h1 = F.lstm(state['c1'], h1_in)
        #h2_in = F.dropout(F.tanh(model.l2_x(h1)), train=train) + model.l2_h(state['h2'])
        #c2, h2 = F.lstm(state['c2'], h2_in)
        #h3 = F.dropout(F.tanh(model.l2_x(h2)), train=train,ratio=0.0)
        if target == False:
            data = h1.data
            data_hidden.append(data)
        
        y = model.l2(h1)
        if target ==False:
            data = y.data
            data_output.append(data)
        state = {'c1': c1, 'h1': h1}
        return state, F.softmax_cross_entropy(y,t)





if __name__ == '__main__':
    #引数読み取り
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='store',dest='data_dir',default='')
    parser.add_argument('--save', action='store',dest='save_filename',default='model.pkl')
    data_dir = parser.parse_args().data_dir
    

    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []

    print "***load data ***"
    #訓練用データのロード
    for x,i in enumerate(i_data):
        for j in xrange(1,8):
            for k in xrange(1,5):
                #元々ファイルがない
                if i == 27 and j == 8 and k == 4:
                    pass
                else:
                    #print eval("'a%d_s%d_t%d_.csv'%(i,j,k)")
                    load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x)

    #テスト用データのロード

    print "***load test ***"
    for x,i in enumerate(i_data):
        for j in xrange(8,9):
            for k in xrange(1,2):
                print eval("'a%d_s%d_t%d_.csv'%(i,j,k)")
                load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)

    state = make_initial_state()

    test_d   = np.array(test_data).astype(np.float32)
    test_t   = np.array(test_target).astype(np.int32)

    #各データの長さ
    N      = len(train_data)
    N_test = len(test_data)
   

    #モデルの初期化
    model = MyChain()
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    model.compute_accuracy = False   

    optimizer = optimizers.SGD(lr=1.,)
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

    
    accum_loss = Variable(np.zeros(()).astype(np.float32)) #明示的にfloat32を指定
    print('going to train {} iterations'.format(jump * n_epoch))

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Learning loop
    for i in xrange(jump * n_epoch):
        # training
        
        x_batch = np.array([train_data[(jump * j+i) % whole_len]
                        for j in six.moves.range(batchsize)]).astype(np.float32)

           
        y_batch = np.array([train_target[(jump * j + i+1) % whole_len]
                        for j in six.moves.range(batchsize)]).astype(np.int32)
        
        
        
        if (i+1) == (jump * n_epoch):
            state, loss = model(x_batch, y_batch, state)
        else:
            state, loss = model(x_batch, y_batch, state)

        accum_loss.data =  accum_loss.data.astype(np.float32)
        accum_loss += loss
        if i %1000 ==0:
            print('epoch = {} \n\ttrain loss: {}'.format(i,accum_loss.data))

        if (i + 1) % bprop_len == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = Variable(mod.zeros(()))
            
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        #テスト
        epoch += 1
            
            #print('loss')
        if (i+1) % (jump*n_epoch) == 0:
            evaluate_loss = evaluate(test_d, test_t,target=False)
        else:
            evaluate_loss = evaluate(test_d, test_t)
        if i % 1000 == 0:
            print('\ttest loss: {}'.format(evaluate_loss))
            
        if i%3000 == 0:
            optimizer.lr /= 1.1
            print('learning rate =', optimizer.lr)
        sys.stdout.flush()

    #chainerの方法を変更
    serializers.save_hdf5('test.model', model)
    serializers.save_hdf5('test.state', optimizer)

