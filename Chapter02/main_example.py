# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
__author__ = 'Alan Hou'

"""
基于 Sklearn 调用 TFIDF
"""

tfidf = TfidfVectorizer()

corpus = ['我 来 到 北京 大学', '他 来到 了 网易 杭研 大厦', '小明 硕士 毕业 于 中国 科学院', '我 爱 北京 天安门'] # 分词后的结果

result = tfidf.fit_transform(corpus).toarray()

print(result)

# 统计关键词
word = tfidf.get_feature_names()
print(word)

# 统计关键词出现的次数
for k,v in tfidf.vocabulary_.items():
    print(k, v)

# 对比第 i 类文本的词语 ti-idf 权重
for i in range(len(result)):
    print('------',i,'------')
    for j in range(len(word)):
        print(word[j], result[i][j])