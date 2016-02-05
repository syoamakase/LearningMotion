# LearningMotion

These programs learn motions, evaluate results, rank what datas is nearest in target data.

##Preparation

 I use [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html) of [データ](http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip) datas.
 

#Requirements

- anaconda

- chainer

- sklearn0.17


##Details

- learning.py : To learn motions (Now, 10(Draw circle), 12(Bowling), 26(Lunge),27(Squat)). It saves results (default: model.pkl).

- evaluate.py : evaluate results of learning.py.

- ranking.py : rank what datas is nearest in target data.