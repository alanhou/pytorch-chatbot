# -*- coding: utf-8 -*-
from nltk.text import TextCollection
__author__ = 'Alan Hou'

corpus = TextCollection(['this is sentence one', 'this is sentence two', 'this is sentence three'])
# 直接算出 tfidf
print(corpus.tf_idf('this', 'this is sentence four'))