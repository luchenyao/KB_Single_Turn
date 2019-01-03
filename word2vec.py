# coding=utf-8

import torch
import itertools
import jieba
from hyperparams import *
from load_data import loadPrepareData


def indexesFromSentence(voc,sentence):
    return [voc.word2index[word] for word in jieba.cut(sentence,cut_all=False)] + [EOS_token]


def zeroPadding(l,fillvalue=PAD_token):
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))


def binaryMatix(l,value=PAD_token):
    m=[]
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


voc,data=loadPrepareData(corpus=corpus,corpus_name=corpus_name,datafile=datafile,save_dir=save_dir)

# # print(data_A[0])
# print(indexes)
# print(num_turns)
# print(real_length)


def inputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    padVar=torch.LongTensor(padList)
    return padVar,lengths


def outputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    max_target_len=max([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    mask=binaryMatix(padList)
    mask=torch.ByteTensor(mask)
    padVar=torch.LongTensor(padList)
    return padVar,mask,max_target_len


def takeFirstLength(elem):
    A=elem.split('\t')
    cut=jieba.cut(A[0],cut_all=False)
    return len(list(cut))

def batch2TrainData(voc,pair_batch):
    pair_batch.sort(key=takeFirstLength,reverse=True)
    # print(pair_batch[:20])
    input_batch,output_batch=[],[]
    for pair in pair_batch:
        pairs=pair.strip().split('\t')
        input_batch.append(pairs[0])
        output_batch.append(pairs[1])

    # print(input_batch[:10])
    # print(output_batch[:10])
    inp,lengths=inputVar(input_batch,voc)
    output,mask,max_target_len=outputVar(output_batch,voc)
    return inp,lengths,output,mask,max_target_len


batch2TrainData(voc,data)



