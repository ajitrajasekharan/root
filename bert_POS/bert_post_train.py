import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer
import nltk
import pdb
import pickle
from pytorch_pretrained_bert import BertModel

from bert_post_common import *




tagged_sents = nltk.corpus.treebank.tagged_sents()
print(len(tagged_sents))
print(tagged_sents[0])
tags = list(set(word_pos[1] for sent in tagged_sents for word_pos in sent))
tags = ["<pad>"] + tags
tags_str = ','.join(tags)
print(len(tags_str))
print(tags_str)
tag2idx = {tag:idx for idx, tag in enumerate(tags)}
idx2tag = {idx:tag for idx, tag in enumerate(tags)}


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(tagged_sents, test_size=.1)
print(len(train_data), len(test_data))
#pdb.set_trace()


device = get_device() 
tokenizer = get_tokenizer()
print(device)


model = Net(vocab_size=len(tag2idx),device = device)
model.to(device)
model = nn.DataParallel(model)


train_dataset = PosDataset(train_data,tokenizer,tag2idx,idx2tag)
eval_dataset = PosDataset(test_data,tokenizer,tag2idx,idx2tag)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=1,
                             collate_fn=pad)
test_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=pad)


optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss(ignore_index=0)

train(model, train_iter, optimizer, criterion)
eval(model, test_iter,tag2idx,idx2tag)


print("Saving model...")
torch.save(model, "my_model/pytorch_model.bin")
print("Model saved")
tags_arr = [tag2idx,idx2tag]
print("Pickling tags...")
fp = open("my_model/tags.pkl","wb")
pickle.dump(tags_arr,fp)
fp.close()
print("Pickling complete...")
#print(open('result', 'r').read().splitlines()[:100])
