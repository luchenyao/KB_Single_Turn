# coding=utf-8

import torch
import re
import random
import unicodedata
import jieba
from io import open
from hyperparams import *


class Voc:
    def __init__(self,name):
        self.name=name
        self.trimmed=False
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3

    def addSentence(self,sentence):
        for word in jieba.cut(sentence,cut_all=False):
            self.addWord(word)
            # print(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.num_words
            self.index2word[self.num_words]=word
            self.word2count[word]=1
            self.num_words+=1
        else:
            self.word2count[word]+=1

    def trim(self,min_count):
        if self.trimmed:
            return
        self.trimmed=True

        keep_words=[]
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        self.word2index={}
        self.word2count={}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words=3

        for word in keep_words:
            self.addWord(word)


# def unicodeToAscii(s):
#     return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')


def normalizeStrings(s):
    # s=unicodeToAscii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s=re.sub(r"\s+",r" ",s).strip()
    return s


def readVocs(datafile,corpus_name):
    print("Reading lines...")
    data=open(datafile,encoding='utf-8').readlines()
    #print(data[:10])
    voc=Voc(corpus_name)
    print(len(data))
    return voc,data


voc,data=readVocs(datafile=datafile,corpus_name=corpus_name)


def filterPairs(data):
    data_filtered = []
    for i in range(len(data)):
        sentences=data[i].strip().split('\t')
        # print(sentences[0])
        # print(sentences[1])
        # print(len(list(jieba.cut(sentences[0].strip()))))
        # print(len(list(jieba.cut(sentences[1].strip()))))
        if (len(list(jieba.cut(sentences[0].strip())))<MAX_LENGTH) and (len(list(jieba.cut(sentences[1].strip())))<MAX_LENGTH):
            data_filtered.append(data[i])
    #print(len(data_filtered))
    return data_filtered

#filterPairs(data)


def loadPrepareData(corpus,corpus_name,datafile,save_dir):
    print("Start preparing training data...")
    voc,data=readVocs(datafile,corpus_name)
    print("Read {!s} turns dialogue".format(len(data)))
    data=filterPairs(data)
    print("Trimmed to {!s} turns dialogue".format(len(data)))
    print("Counting words...")
    for sentences in data:
        for sentence in sentences.split('\t'):
            voc.addSentence(sentence)
    print("Counted words:",voc.num_words)
    return voc,data


# def trimRareWords(voc,data_A,data_B,MIN_COUNT):
#     voc.trim(MIN_COUNT)
#     keep_pairs=[]
#     for sentences in data_A:
#         input_sentence=sentences.split('\t')
#         keep_input=True
#         for i in input_sentence:
#             for word in jieba.cut(sentence,cut_all=False):
#                 if word not in voc.word2index:
#                     keep_input=False
#                     break
#         for word in jieba.cut(output_sentence,cut_all=True):
#             if word not in voc.word2index:
#                 keep_output=False
#                 break
#         if keep_input and keep_output:
#             keep_pairs.append(pair)
#
#     print("Trimmed from {} pairs to {},{:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
#     return keep_pairs


# voc,data_A,data_B=loadPrepareData(corpus=corpus,corpus_name=corpus_name,datafile_A=datafile_A,datafile_B=datafile_B,save_dir=save_dir)
# # pairs=trimRareWords(voc,pairs,MIN_COUNT)
# batchs=batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
# input_variable,lengths,target_variable,mask,max_target_len=batchs
#
# print("input_variable:",input_variable)
# print("lengths:",lengths)
# print("target_variable:",target_variable)
# print("mask:",mask)
# print("max_target_len:",max_target_len)





