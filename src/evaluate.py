#!/usr/bin/env python

import argparse
import sys
import time
import numpy as np
import six
from chainer import cuda, Variable, FunctionSet, optimizers, serializers, Chain
import chainer.functions  as F
import csv
import pickle
import cPickle
import matplotlib.pyplot as plt
import argparse
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report,hamming_loss,f1_score,roc_curve,confusion_matrix

import mynet
import mynet_not_lstm
import mynet_cnn

plt.style.use('ggplot')

mod = np

batchsize = 540
n_units   = 120
# i_data = [10,12,22,23,26,27]
i_data = [col for col in xrange(1,28)]
classnum  = len(i_data) 

test_data   = []
test_target = []
true_matrix = []

def load_csv(data_dir,data_file_name,num,test=False):
    data = []
    target = []
    with open(data_dir+"/"+data_file_name,"rU") as f:
        data_file = csv.reader(f,delimiter=",")
        for i, d in enumerate(data_file):
            data.append(np.array(d[1:]).astype(np.float32))
            target.append([num])
        
    if test:
        test_data.extend(data)
        test_target.extend(target)
        true_matrix.append(num)
    return

def printscore(true_matrix,max_matrix,voting_matrix,mean_matrix):

    print "\t\tvoting classification report"
    print classification_report(true_matrix,voting_matrix)
    print "\t\tmax classification report"
    print classification_report(true_matrix,max_matrix)
    print "\t\tmean classification report"
    print classification_report(true_matrix,mean_matrix)

    print "------------------------------------------"

    print "\tvoting accuracy :{}\n".format(accuracy_score(true_matrix,voting_matrix))
    print "max accuracy :{}\n".format(accuracy_score(true_matrix,max_matrix))
    print "mean accuracy :{}\n".format(accuracy_score(true_matrix,mean_matrix))

    print "------------------------------------------"

    print "\tvoting Hamming loss:{}\n".format(hamming_loss(true_matrix,voting_matrix))
    print "max Hamming loss:{}\n".format(hamming_loss(true_matrix,max_matrix))
    print "mean Hamming loss:{}\n".format(hamming_loss(true_matrix,mean_matrix))

    print "------------------------------------------"

    print "\tvoting f1 score:{}\n".format(f1_score(true_matrix,voting_matrix,average='macro'))
    print "max f1 score:{}\n".format(f1_score(true_matrix,max_matrix,average='macro'))
    print "mean f1 score:{}\n".format(f1_score(true_matrix,mean_matrix,average='macro'))

    print "------------------------------------------"

    fpr, tpr, thresholds = roc_curve(true_matrix, voting_matrix, pos_label=2)
    print "\tvoting auc:{}\n".format(metrics.auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(true_matrix, max_matrix, pos_label=2)
    print "max auc:{}\n".format(metrics.auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(true_matrix, mean_matrix, pos_label=2)
    print "mean auc:{}\n".format(metrics.auc(fpr, tpr))


    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='store',dest='data_dir',default='')
    parser.add_argument('--load', action='store',dest='load_filename',default='test')

    data_dir      = parser.parse_args().data_dir
    load_filename = parser.parse_args().load_filename
    print "***load test ***"

    mean_matrix   = []
    max_matrix    = []
    voting_matrix = []

    for x,i in enumerate(i_data):
        for j in xrange(1,8):
            for k in xrange(1,5):
                if (i == 23 and j == 6 and k == 4) or (i==8 and j == 1 and k==4) or (i == 27 and j == 8 and k == 4):
                    continue
                print eval("'a%d_s%d_t%d_.csv'%(i,j,k)")
                test_data   = []
                test_target = []
                load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)

                test_d   = np.array(test_data).astype(np.float32)
                test_t   = np.array(test_target).astype(np.int32)

                N_test = len(test_data)

                model = mynet_cnn.MyChain(n_units,classnum,batchsize)
                model.compute_accuracy = False

                optimizer = optimizers.RMSprop()
                optimizer.setup(model)
                serializers.load_hdf5(load_filename+'.model', model)
                serializers.load_hdf5(load_filename+'.state',optimizer)
                
                def make_initial_state(batchsize=batchsize, train=True):
                    return {name: Variable(mod.zeros((batchsize, n_units),
                                                             dtype=np.float32),
                                                   volatile=not train)
                            for name in ('c1', 'h1','c2','h2')}

                def evaluate(x_data, t,target=True):
                    state = make_initial_state(batchsize=len(t), train=False)
                    state, loss = model(x_data, t, state, train=False,target=target)
                    return loss.data.astype(np.float32)

                state = make_initial_state(batchsize=len(test_data))
                accum_loss = Variable(np.zeros(()).astype(np.float32))

                model.data_output = []
                evaluate_loss = evaluate(test_d, test_t,target=False)
                print('\ttest loss: {}'.format(evaluate_loss))

                data_output_exp = np.exp(model.data_output)
                result = data_output_exp/data_output_exp.sum(axis=1)
                mean_top = result.mean(axis=1).argmax()
                max_top  = result.max(axis=1).argmax()
                mean_matrix.append(mean_top)
                max_matrix.append(max_top)

                result = np.array(model.data_output).astype(np.float32)
                arr = np.zeros(len(i_data)).astype(np.int32)
                count = 0
                for d in result[0]:
                    arr[d.argmax()] +=1 
                voting_matrix.append(arr.argmax())


    target_names = i_data

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm_max = confusion_matrix(true_matrix, max_matrix)
    cm_mean = confusion_matrix(true_matrix, mean_matrix)
    cm_voting = confusion_matrix(true_matrix, voting_matrix)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cm_max,title='Max confusion matrix')
    plt.show()
    plot_confusion_matrix(cm_mean,title='Mean confusion matrix')
    plt.show()
    plot_confusion_matrix(cm_voting,title='Voting confusion matrix')
    plt.show()
    '''
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    '''
    printscore(true_matrix,max_matrix,voting_matrix,mean_matrix)

    cm_normalized = cm_voting.astype('float') / cm_voting.sum(axis=1)[:, np.newaxis]
    print cm_normalized


