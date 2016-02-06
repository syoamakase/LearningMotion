# LearningMotion

These programs learn motions, evaluate results, rank what datas is nearest in target data.

##Preparation

 I use [datas](http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip) in [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html)

##Installation

`pip install -r requirements.txt`

if your machine is MAC, create `matplotlibrc` in ~/.matplotlib and wirte below to write graph.

`backend : TkAgg` 


##Details

- learning.py : To learn motions (Now, 10(Draw circle), 12(Bowling), 26(Lunge),27(Squat)). It saves results (default: model.pkl).

- evaluate.py : evaluate results of learning.py.

- ranking.py : rank what datas is nearest in target data.