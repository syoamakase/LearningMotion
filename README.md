# LearningMotion

These programs learn motions, evaluate results, rank what datas is nearest in target data.

##Preparation

 I use [datas](http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip) in [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html)
 
 

##Installation

`pip install -r requirements.txt`

And,python version is 2.7.9 .


if your machine is MAC, create `matplotlibrc` in ~/.matplotlib and wirte below to write graph.



`backend : TkAgg` 


##Details

- learning.py : 

 To learn motions. It saves results (default: 'test.state' and 'test.model').

 ```
 usage: learningData.py [-h] [--dir DATA_DIR] [--save SAVE_FILENAME]
optional arguments:
  -h, --help            show this help message and exit
  --dir DATA_DIR
  --save SAVE_FILENAME
  
 ```
 
 If you want to change data, change `i_data` in learning.py


- evaluate.py :   
   Evaluate results of learning.py. It uses accuracy, f1-score, auc, hamming loss ,and make confusion matrix.
	

- ranking.py : rank what datas is nearest in target data.