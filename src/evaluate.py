#! usr/bin/python
#-*-coding:utf-8 -*- 

# 必要モジュールのインポート
import argparse
import sys
import time
import numpy as np
import six
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import csv
import pickle
import cPickle
import matplotlib.pyplot as plt
import argparse
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report,hamming_loss,f1_score,roc_curve,confusion_matrix

plt.style.use('ggplot')

mod = np

batchsize = 40
n_units = 250

test_data   = []
test_target = []
true_matrix = []


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
                #data.append(np.array(d[1:]).astype(np.float32)-prev_data[1:])
                #prev_data= np.array(d).astype(np.float32)
                data.append(np.array(d[1:]).astype(np.float32))
                target.append([num])
                data_length = i
        
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
    parser.add_argument('--load', action='store',dest='load_filename',default='model.pkl')

    data_dir      = parser.parse_args().data_dir
    load_filename = parser.parse_args().load_filename
    #テスト用データのロード
    print "***load test ***"
    i_data = [10,12,26,27]

    mean_matrix   = []
    max_matrix    = []
    voting_matrix = []

    for x,i in enumerate(i_data):
        for j in xrange(8,9):
            for k in xrange(1,5):
                #元々ファイルがない
                if i == 27 and j == 8 and k == 4:
                    continue
                print eval("'a%d_s%d_t%d_.csv'%(i,j,k)")
                #print x
                test_data   = []
                test_target = []
                load_csv(data_dir,eval("'a%d_s%d_t%d_.csv'%(i,j,k)"),x,test=True)


                test_d   = np.array(test_data).astype(np.float32)
                test_t   = np.array(test_target).astype(np.int32)

                N_test = len(test_data)


                #学習済みモデルのロード
                model = pickle.load(open(data_dir+load_filename,'rb'))

                data_output = []
                data_hidden = []
                data_first  = []
                def forward_one_step(x_data, y_data, state, train=True,target=True):
                    x = Variable(x_data, volatile=not train)
                    t = Variable(y_data.flatten(), volatile=not train)
                    #print t.data
                    h0 = model.l0(x)
                    if target == False:
                        data = h0.data
                        data_first.append(data)

                    h1_in = F.tanh(model.l1_x(h0)) + model.l1_h(state['h1'])
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

                def make_initial_state(batchsize=batchsize, train=True):
                    return {name: Variable(mod.zeros((batchsize, n_units),
                                                             dtype=np.float32),
                                                   volatile=not train)
                            for name in ('c1', 'h1')}


                def evaluate(x_data, t,target=True):
                    state = make_initial_state(batchsize=len(t), train=False)
                    state, loss = forward_one_step(x_data, t, state, train=False,target=target)
                    return loss.data.astype(np.float32)


                state = make_initial_state(batchsize=len(test_data))
                accum_loss = Variable(np.zeros(()).astype(np.float32)) #明示的にfloat32を指定


                evaluate_loss = evaluate(test_d, test_t,target=False)
                #print('data length: {}'.format(N_test))
                print('\ttest loss: {}'.format(evaluate_loss))
                #result = np.array(data_output).astype(np.float32)
                data_output_exp = np.exp(data_output)
                result = data_output_exp/data_output_exp.sum(axis=1)
                #print result
                mean_top = result.mean(axis=1).argmax()
                max_top  = result.max(axis=1).argmax()
                mean_matrix.append(mean_top)
                max_matrix.append(max_top)
                result = np.array(data_output).astype(np.float32)
                arr = np.zeros(5).astype(np.int32)
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


