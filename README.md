This BERT based POS tagger is derived almost entirely on https://github.com/Kyubyong ipython notebook

https://github.com/Kyubyong/nlp_made_easy

**Requirements**

- Pytorch (conda install -c pytorch pytorch)
- Pytorch pre-trained BERT . (pip install pytorch-pretrained-bert)
- nltk (pip install nltk)


**Data**

- Automatically fetches treenbank training data using nltk (when running first time, it will prompt you to install treebank. This can from within python prompt in command line)


**Usage**

1. Create a directory my_model

2. python bert_post_train.py . (this will result in training with accuracy 98%)

3. To test. python bert_post_test.py
