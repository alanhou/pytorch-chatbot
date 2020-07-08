# -*- coding: utf-8 -*-
import gzip
import  gensim
import logging
__author__ = 'Alan Hou'

logging.basicConfig(format="", level=logging.INFO)

# 解压缩数据
data_file = "reviews_data.txt.gz"

# 把文件读入至 list
def read_input(input_file):
    logging.info("reading file {0} ... this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i,line in enumerate(f):
            if (i%10000==0):
                logging.info("read {0} reviews".format(i))
            yield gensim.utils.simple_preprocess(line)

documents = list(read_input(data_file))
logging.info("Done reading data file")

# 训练 model
model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
model.train(documents, total_examples=len(documents), epochs=10)

w1 = "dirty"
result = model.wv.most_similar(positive=w1)

print(result)

"""
[
    ('filthy', 0.870586633682251), 
    ('stained', 0.7817946672439575), 
    ('dusty', 0.7699412107467651), 
    ('unclean', 0.7630631923675537), 
    ('grubby', 0.756673276424408), 
    ('smelly', 0.7458912134170532), 
    ('grimy', 0.7415663003921509), 
    ('dingy', 0.7353354692459106), 
    ('soiled', 0.7284513711929321), 
    ('disgusting', 0.724179208278656)
]
"""