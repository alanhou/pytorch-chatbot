# -*- coding: utf-8 -*-
import jieba.posseg as pseg
__author__ = 'Alan Hou'

words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print(f"{word} {flag}")

# 我 r       # 代词
# 爱 v       # 动词
# 北京 ns     # 名词
# 天安门 ns    # 名词