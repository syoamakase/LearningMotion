#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要モジュールのインポート
import argparse
import sys
import time
import numpy as np
import six
from chainer import cuda, Variable, FunctionSet, optimizers, Chain,serializers
import chainer.functions  as F
import csv
import pickle
import cPickle
import matplotlib.pyplot as plt

plt.style.use('ggplot')

mod = np

batchsize = 40
n_units = 250

test_data   = []
test_target = []

data_output = []
data_hidden = []
data_first  = []

target_data = []

#検索データ([action,subject,take])
search_data = {"action":12,"subject":1,"take":1}
subect_name = 8
i_data =[10,12,26,27]

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
                data.append(np.array(d[1:]).astype(np.float32))
                target.append([num])
                data_length = i
        
    if test:
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
                l2=F.Linear(n_units,4),
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
        else:
        	data = y.data
        	target_data.append(data)

        state = {'c1': c1, 'h1': h1}
        return state, F.softmax_cross_entropy(y,t)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', action='store',dest='data_dir',default='')
	parser.add_argument('--load', action='store',dest='load_filename',default='model.pkl')

	data_dir = parser.parse_args().data_dir
	load_filename = parser.parse_args().load_filename

	
	

	#元々ファイルがない
	if search_data["action"] == 27 and search_data["subject"] == 8 and search_data["take"] == 4:
		pass
	else:
		#print eval("'a%d_s%d_t%d_.csv'%(search_data.action,search_data.subject,search_data.take)")
		load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(search_data['action'],search_data['subject'],search_data['take'])"),2,test=True)

	test_d   = np.array(test_data).astype(np.float32)
	test_t   = np.array(test_target).astype(np.int32)

	N_test = len(test_data)
	model = MyChain()
	model.compute_accuracy = False
	optimizer = optimizers.SGD(lr=1.)
	optimizer.setup(model)
	serializers.load_hdf5('test.model',model)
	serializers.load_hdf5('test.state',optimizer)


	def evaluate(x_data, t,target=True):
		state = make_initial_state(batchsize=len(t), train=False)
		state, loss = model(x_data, t, state, train=False,target=target)
					
		return loss.data.astype(np.float32)

	evaluate_loss = evaluate(test_d, test_t,target=True)
	#print data_output
	data_output_exp = np.exp(target_data)
	target_result = data_output_exp/data_output_exp.sum(axis=1)

	dist_data = []
	for x,i in enumerate(i_data):
		for j in xrange(8,9):
			for k in xrange(1,5):
				#元々ファイルがない
				if i == 27 and j == 8 and k == 4:
					continue

				#print x
				test_data   = []
				test_target = []
				load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)

				test_d   = np.array(test_data).astype(np.float32)
				test_t   = np.array(test_target).astype(np.int32)
				N_test = len(test_data)
				#data_output = []
				data_output = []
				evaluate_loss = evaluate(test_d, test_t,target=False)
				#print('\ttest loss: {}'.format(evaluate_loss))
				data_output_exp = np.exp(data_output)
				result = data_output_exp/data_output_exp.sum(axis=1)
				distance = (target_result[0][0][0:20] - result[0][0][0:20])**2
				dist_data.append(distance.sum())

	d = dist_data[:]
	print '***ranking***'
	print("target: a%d_s%d_t%d"%(search_data["action"],search_data["subject"],search_data["take"]))
	for i in range(len(d)):
		sys.stdout.write("%d: "%(i+1))
		min = np.argmin(d)
		if min < 4:
			print("a%d_s%d_t%d(dist:%f)"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))
		elif min < 8:
			print("a%d_s%d_t%d(dist:%f)"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))
		elif min < 12:
			print("a%d_s%d_t%d(dist:%f)"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))
		else:
			print("a%d_s%d_t%d(dist:%f)"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))
		d[min] = float("inf")