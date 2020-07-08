# -*- coding: utf-8 -*-
import nltk
# nltk.download('averaged_perceptron_tagger')
__author__ = 'Alan Hou'

text = nltk.word_tokenize("And now for something completely different")
# 词性标注
pos_tags = nltk.pos_tag(text)
print(pos_tags)

# [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]