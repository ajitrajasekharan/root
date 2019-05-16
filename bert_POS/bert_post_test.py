import os
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


device = get_device() 
tokenizer = get_tokenizer()

print("Loading model ...")
model = torch.load("my_model/pytorch_model.bin")
print("Loading model complete")
print("Loading Pickling tags...")
fp = open("my_model/tags.pkl","rb")
tags_arr = pickle.load(fp)
print("Loading Pickling tags complete")
fp.close()


while True:
	print("Enter text:")
	text = input()
	rt_test_dataset = PosDataset(construct_input(text),tokenizer,tags_arr[0],tags_arr[1])
	rt_test_iter = data.DataLoader(dataset=rt_test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=pad)


	ret_arr = test(model, rt_test_iter,tags_arr[0],tags_arr[1])
	print(ret_arr)

#print(open('result', 'r').read().splitlines()[:100])
