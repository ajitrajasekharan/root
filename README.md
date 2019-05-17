This BERT based POS tagger is derived almost entirely from https://github.com/Kyubyong ipython notebook

https://github.com/Kyubyong/nlp_made_easy

**Requirements**

- Pytorch (conda install -c pytorch pytorch)
- Pytorch pre-trained BERT . (pip install pytorch-pretrained-bert)
- nltk (pip install nltk)


**Data**

- Automatically fetches treebank training data using nltk _(when running first time, it will prompt  to install treebank. This can be done from within python prompt in command line)_


**Usage**

1. To train. _python bert_post_train.py < model dir to save. e.g. out >_ . (this will result in training with accuracy ~98%)

3. To test. _python bert_post_test.py < model dir to load >_

    __Example input:__ _The bird flew over the house and perched on a tree_

    __Output__:   _[('The', 'DT'), ('bird', 'NN'), ('flew', 'VBD'), ('over', 'IN'), ('the', 'DT'), ('house', 'NN'), ('and', 'CC'), ('perched', 'VBD'), ('on', 'IN'), ('a', 'DT'), ('tree', 'NN')]_



**License**

MIT License
