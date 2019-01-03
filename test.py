# coding=utf-8

import torch
import torch.nn as nn
from encoder import EncoderRNN
from attn_deocder import LuongAttnDecoderRNN
from hyperparams import *
from load_data import loadPrepareData
# voc,pairs=loadPrepareData(corpus,corpus_name,datafile,save_dir)
# pairs=trimRareWords(voc,pairs,MIN_COUNT)
# print(voc.num_words)
# embedding=nn.Embedding(voc.num_words,hidden_size)
# print(embedding)
# encoder=EncoderRNN(hidden_size,embedding,encoder_n_layers,dropout)
# decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,output_size,decoder_n_layers,dropout)


def takeSecond(elem):
    return len(elem)


# 列表
random = [[2, 2,4], [3, 4], [4], [1, 3]]
# 指定第二个元素排序
random.pop(0)
random.pop(1)

# 输出类别
print("排序列表", random)