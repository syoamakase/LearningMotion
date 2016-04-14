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

import mynet
import mynet_not_lstm
import mynet_cnn

plt.style.use('ggplot')

mod = np

batchsize = 540
n_units = 120

test_data   = []
test_target = []

data_output = []
data_hidden = []
data_first  = []

target_data = []

#検索データ([action,subject,take])
search_data = {"action":7,"subject":5,"take":3}
subect_name = 8
i_data = [col for col in xrange(1,28)]
classnum = len(i_data)

# csvファイルを読み込む関数
def load_csv(data_dir,data_file_name,num,test=False):
    data = []
    target = []
    with open(data_dir+"/"+data_file_name,"rU") as f:
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
	for name in ('c1', 'h1','c2','h2')}



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
	model = mynet_cnn.MyChain(n_units,classnum,batchsize)
	model.compute_accuracy = False
	optimizer = optimizers.SGD(lr=1.)
	#optimizer  = optimizers.AdaGrad()
	optimizer.setup(model)
	serializers.load_hdf5('sample.model',model)
	serializers.load_hdf5('sample.state',optimizer)

	def evaluate(x_data, t,target=True):
		state = make_initial_state(batchsize=len(t), train=False)
		state, loss = model(x_data, t, state, train=False,target=target)
					
		return loss.data.astype(np.float32)

	evaluate_loss = evaluate(test_d, test_t,target=False)
	#print data_output
	data_output_exp = np.exp(model.data_output)
	target_result = data_output_exp/data_output_exp.sum(axis=1)

	dist_data = []
	for x,i in enumerate(i_data):
		for j in xrange(8,9):
			for k in xrange(1,5):
				#元々ファイルがない
				if (i == 23 and j == 6 and k == 4) or (i==8 and j == 1 and k==4) or (i == 27 and j == 8 and k == 4):
					continue

				#print x
				test_data   = []
				test_target = []
				load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)

				test_d   = np.array(test_data).astype(np.float32)
				test_t   = np.array(test_target).astype(np.int32)
				N_test = len(test_data)

				#data_output = []
				model.data_output = []
				evaluate_loss = evaluate(test_d, test_t,target=False)
				#print('\ttest loss: {}'.format(evaluate_loss))
				data_output_exp = np.exp(model.data_output)
				result = data_output_exp/data_output_exp.sum(axis=1)
				distance = (target_result[0][0][0:20] - result[0][0][0:20])**2
				dist_data.append(distance.sum())

	d = dist_data[:]
	print '***ranking***'
	print("target: a%d_s%d_t%d"%(search_data["action"],search_data["subject"],search_data["take"]))
	for i in range(len(d)):
		sys.stdout.write("%d: "%(i+1))
		min = np.argmin(d)
		if (min//4+1) == search_data["action"]:
			print("\033[91ma%d_s%d_t%d(dist:%f)\033[0m"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))
		else:
			print("a%d_s%d_t%d(dist:%f)"%(i_data[min//4],subect_name,min%4+1,dist_data[min]))

		d[min] = float("inf")
