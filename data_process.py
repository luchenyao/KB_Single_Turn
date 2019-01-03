# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script,trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open


def printLines(file,n=10):
    with open(file,'rb') as datafile:
        lines=datafile.readlines()
    for line in lines[:n]:
        print(line)


# printLines(os.path.join(corpus,"movie_lines.txt"))


def loadLines(fileName,fields):
    lines={}
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values=line.split(' +++$+++ ')
            lineObj={}
            for i,field in enumerate(fields):
                lineObj[field]=values[i]
            # 嵌套字典
            lines[lineObj['lineID']]=lineObj
    return lines


def loadConversations(fileName,lines,fields):
    conversations=[]
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values=line.split(' +++$+++ ')
            convObj={}
            for i,field in enumerate(fields):
                convObj[field]=values[i]

            lineIds=eval(convObj["utteranceIDs"])
            convObj["lines"]=[]
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractSentencePairs(conversations):
    qa_pairs=[]
    for conversation in conversations:
        for i in range(len(conversation["lines"])-1):
            inputLine=conversation["lines"][i]["text"].strip()
            targetLine=conversation["lines"][i+1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine,targetLine])
    return qa_pairs
















